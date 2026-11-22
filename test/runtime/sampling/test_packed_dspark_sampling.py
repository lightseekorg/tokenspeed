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

"""Runtime wiring for adaptive DSpark's packed probability sampler."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=90, suite="runtime-1gpu", disabled_on_runners=["amd-*"])

from tokenspeed_kernel.ops.sampling.packed_verify import (  # noqa: E402
    DRAFT_POS_HASH_MUL,
    RNG_TAG_DRAFT,
    sample_probs,
    verify_packed_chain,
)
from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    get_apply_vocab_mask_func,
)
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.flashinfer import (  # noqa: E402
    FlashInferSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="adaptive DSpark sampling requires NVIDIA CUDA",
)

VOCAB, POOL, WIDTH = 32, 8, 4


@pytest.fixture(autouse=True, params=[32, 129280])
def vocabulary(request, monkeypatch):
    monkeypatch.setitem(globals(), "VOCAB", request.param)


def _params(seed: int, temperature: float, top_k: int, top_p: float) -> SamplingParams:
    params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )
    params.resolve_seed(f"packed_dspark_{seed}")
    params.normalize(None)
    return params


def _backend(
    monkeypatch,
    max_bs: int = 4,
    *,
    reject_threshold: float = 2.0,
    output_logprobs: bool = False,
) -> FlashInferSamplingBackend:
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", True)
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", "standard"
    )
    return FlashInferSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="per-request",
            logprob_order="torch",
            enable_output_logprobs=output_logprobs,
            max_bs=max_bs,
            max_draft_tokens_per_req=WIDTH,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
            spec_reject_draft_prob_threshold=reject_threshold,
        )
    )


def _allow_tokens(allowed: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(
        (allowed.shape[0], (VOCAB + 31) // 32),
        dtype=torch.int32,
        device=allowed.device,
    )
    rows = torch.arange(allowed.shape[0], device=allowed.device)
    for choice in range(allowed.shape[1]):
        tokens = allowed[:, choice].long()
        bits = (torch.ones_like(tokens) << (tokens % 32)).to(torch.int32)
        mask[rows, tokens // 32] |= bits
    return mask


def _apply_vocab_mask_torch(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    bits = torch.arange(32, device=logits.device)
    allowed = (vocab_mask.long().unsqueeze(-1) >> bits) & 1
    allowed = allowed.flatten(1).bool()[:, : logits.shape[1]]
    logits.masked_fill_(~allowed, float("-inf"))


def _vocab_mask_applier():
    return get_apply_vocab_mask_func("xgrammar")


def _masked_reference(logits: torch.Tensor, vocab_mask: torch.Tensor) -> torch.Tensor:
    reference = logits.clone()
    _apply_vocab_mask_torch(reference, vocab_mask)
    return reference


def _assert_packed_logprobs(
    actual: torch.Tensor,
    logits: torch.Tensor,
    vocab_mask: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Independently map flat request/column indices through the packed CSR."""
    starts = query_start_loc.tolist()
    rows = []
    for request, indices in enumerate(accept_index.tolist()):
        for index in indices:
            column = index - request * WIDTH if index >= 0 else 0
            rows.append(starts[request] + column)
    row_indices = torch.tensor(rows, dtype=torch.int64, device=logits.device)
    reference = _masked_reference(logits, vocab_mask).index_select(0, row_indices)
    expected = torch.log_softmax(reference, dim=-1, dtype=torch.float32).gather(
        1, predict.long().unsqueeze(1)
    )
    torch.testing.assert_close(actual, expected.squeeze(1), atol=1e-6, rtol=1e-6)


def _grammar_rows(rows: int) -> torch.Tensor:
    requests = torch.arange(rows, device="cuda")
    first = (requests * 5 + 3) % VOCAB
    second = (first + 1 + requests % 7) % VOCAB
    # Include a high vocabulary lane while retaining at least two allowed IDs
    # per row, so selected-token logprobs remain sensitive to row alignment.
    allowed = torch.stack((first, second, torch.full_like(first, VOCAB - 1)), dim=1)
    return _allow_tokens(allowed)


def _prepare(backend, rids, slots, params) -> torch.Tensor:
    backend.prepare_step(
        admission_ids=rids,
        request_pool_indices=slots,
        sampling_params_list=params,
        num_tokens_per_req=WIDTH,
    )
    return torch.tensor(slots, device="cuda", dtype=torch.int64)


def _logits(rows: int) -> torch.Tensor:
    columns = torch.arange(VOCAB, device="cuda", dtype=torch.float32)
    result = []
    for row in range(rows):
        center = 4.0 + (row * 7 % (VOCAB - 8))
        result.append(-((columns - center) ** 2) / (3.0 + row % 4) + columns * 1e-3)
    return torch.stack(result).contiguous()


def _reference_probs(
    logits: torch.Tensor,
    params_by_slot: dict[int, SamplingParams],
    row_slots: list[int],
) -> torch.Tensor:
    rows = []
    for row, slot in enumerate(row_slots):
        params = params_by_slot[slot]
        probs = torch.softmax(logits[row].float() / params.temperature, dim=-1)
        if params.top_k < VOCAB:
            ids = torch.topk(probs, params.top_k).indices
            selected = torch.zeros_like(probs)
            selected[ids] = probs[ids]
            probs = selected / selected.sum()
        if params.top_p < 1.0:
            values, ids = torch.sort(probs, descending=True, stable=True)
            keep = min(int((values.cumsum(0) < params.top_p).sum()) + 1, VOCAB)
            selected = torch.zeros_like(probs)
            selected[ids[:keep]] = probs[ids[:keep]]
            probs = selected / selected.sum()
        rows.append(probs)
    return torch.stack(rows)


def _assert_probabilities(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual.sum(-1), torch.ones(actual.shape[0], device="cuda"), atol=2e-6, rtol=0
    )


def test_dspark_greedy_ties_record_a_single_token(monkeypatch):
    backend = _backend(monkeypatch)
    slots = _prepare(backend, [101], [5], [_params(991, 1.0, 1, 1.0)])
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    probabilities = backend.sample_dspark_proposals(
        torch.zeros((1, VOCAB), device="cuda"),
        slots,
        torch.tensor([17], device="cuda", dtype=torch.int64),
        output,
    )
    assert output.item() == 0
    assert probabilities[0, 0].item() == 1.0
    assert probabilities.count_nonzero().item() == 1


def _verify(
    backend,
    *,
    slots: torch.Tensor,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    draft_probs: torch.Tensor,
    query_start_loc: torch.Tensor,
    frontiers: torch.Tensor,
    vocab_mask: torch.Tensor | None = None,
):
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=frontiers,
        draft_probs=draft_probs,
        vocab_size=VOCAB,
        vocab_mask=vocab_mask,
        apply_vocab_mask=_vocab_mask_applier() if vocab_mask is not None else None,
        device="cuda",
    )
    output = LogitsProcessorOutput(next_token_logits=logits)
    predict, lengths = backend.verify(
        output,
        info,
        candidates,
        tree=None,
        query_start_loc=query_start_loc,
    )
    indices = backend._accept_index_buf[: candidates.numel()].view_as(candidates)
    if output.next_token_logprobs is not None:
        return predict, lengths, indices, output.next_token_logprobs
    return predict, lengths, indices


def _make_verify_case(backend, *, starts=None, frontiers=None):
    slots = torch.tensor([5, 2, 7], device="cuda", dtype=torch.int64)
    rids = [101, 102, 103]
    params = [
        _params(1105, 0.7, 7, 0.82),
        _params(2202, 1.15, 5, 0.71),
        _params(7707, 0.9, 9, 0.91),
    ]
    params_by_slot = dict(zip(slots.tolist(), params, strict=True))
    _prepare(backend, rids, slots.tolist(), params)

    if starts is None:
        starts = [0, 4, 6, 7]
    query_start_loc = torch.tensor(starts, device="cuda", dtype=torch.int64)
    if frontiers is None:
        frontiers = torch.zeros(POOL + 1, device="cuda", dtype=torch.int32)
        frontiers[5], frontiers[2], frontiers[7] = 300, 50, 900
    logits = _logits(int(starts[-1]))
    draft_probs = torch.full(
        (POOL + 1, WIDTH, VOCAB),
        backend.config.spec_reject_draft_prob_threshold + 1.0,
        device="cuda",
        dtype=torch.float32,
    )

    # Record three proposal laws for slot 5 by sampling from logits identical
    # to its first three target rows. The full-width request must accept them.
    proposal_logits = logits[:3].contiguous()
    proposal_slots = slots[:1].expand(3).contiguous()
    proposal_positions = (
        frontiers[5].to(torch.int64)
        + 1
        + (torch.arange(3, device="cuda", dtype=torch.int64) + 1) * DRAFT_POS_HASH_MUL
    )
    proposal_tokens = torch.empty(3, device="cuda", dtype=torch.int32)
    proposal_probs = backend.sample_dspark_proposals(
        proposal_logits, proposal_slots, proposal_positions, proposal_tokens
    )
    draft_probs[5, :3].copy_(proposal_probs)
    candidates = torch.tensor(
        [[VOCAB - 1, 0, 0, 0], [VOCAB - 1, 0, 0, 0], [VOCAB - 1, 0, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    candidates[0, 1:] = proposal_tokens
    return (
        slots,
        params_by_slot,
        logits,
        candidates,
        draft_probs,
        query_start_loc,
        frontiers,
    )


def _direct_verify_reference(
    backend,
    *,
    slots: torch.Tensor,
    params_by_slot: dict[int, SamplingParams],
    logits: torch.Tensor,
    candidates: torch.Tensor,
    draft_probs: torch.Tensor,
    query_start_loc: torch.Tensor,
    frontiers: torch.Tensor,
):
    counts = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    row_slots = [
        int(slots[request])
        for request, count in enumerate(counts)
        for _ in range(count)
    ]
    target_probs = backend._dspark_probabilities(
        logits, torch.tensor(row_slots, device="cuda", dtype=torch.int64)
    )
    seeds = torch.tensor(
        [params_by_slot[int(slot)].seed for slot in slots.tolist()],
        device="cuda",
        dtype=torch.int64,
    )
    positions = frontiers[slots].to(torch.int64) + 1
    predict = torch.empty(candidates.numel(), device="cuda", dtype=torch.int32)
    indices = torch.empty_like(candidates)
    lengths = torch.empty(slots.numel(), device="cuda", dtype=torch.int32)
    verify_packed_chain(
        reject_draft_prob_threshold=backend.config.spec_reject_draft_prob_threshold,
        predicts=predict,
        accept_index=indices,
        accept_length=lengths,
        candidates=candidates,
        target_probs=target_probs,
        draft_probs=draft_probs,
        draft_pool_indices=slots,
        query_start_loc=query_start_loc,
        seeds=seeds,
        positions=positions,
    )
    return predict, lengths, indices, target_probs


def test_dspark_proposals_record_filtered_probabilities_and_follow_slots(monkeypatch):
    backend = _backend(monkeypatch, max_bs=3)
    slots = torch.tensor([5, 2, 7], device="cuda", dtype=torch.int64)
    params = [
        _params(101, 0.55, 4, 0.73),
        _params(202, 1.1, 8, 0.91),
        _params(707, 0.8, 3, 0.66),
    ]
    _prepare(backend, [101, 102, 103], slots.tolist(), params)
    params_by_slot = dict(zip(slots.tolist(), params, strict=True))
    logits = _logits(3)
    positions = torch.tensor([11, 202, 707], device="cuda", dtype=torch.int64)
    output = torch.empty(3, device="cuda", dtype=torch.int32)
    actual_probs = backend.sample_dspark_proposals(logits, slots, positions, output)
    expected_probs = _reference_probs(logits, params_by_slot, slots.tolist())
    _assert_probabilities(actual_probs, expected_probs)

    expected_tokens = torch.empty_like(output)
    sample_probs(
        probs=actual_probs,
        seeds=torch.tensor(
            [param.seed for param in params], device="cuda", dtype=torch.int64
        ),
        positions=positions,
        output=expected_tokens,
        rng_tag=RNG_TAG_DRAFT,
    )
    torch.testing.assert_close(output, expected_tokens, atol=0, rtol=0)

    permutation = torch.tensor([2, 0, 1], device="cuda")
    reordered_tokens = torch.empty_like(output)
    reordered_probs = backend.sample_dspark_proposals(
        logits[permutation].contiguous(),
        slots[permutation].contiguous(),
        positions[permutation].contiguous(),
        reordered_tokens,
    )
    _assert_probabilities(reordered_probs, actual_probs[permutation])
    torch.testing.assert_close(reordered_tokens, output[permutation], atol=0, rtol=0)


@pytest.mark.parametrize("reject_threshold", [1.0, 2.0])
def test_dspark_verify_maps_packed_rows_and_matches_kernel(
    monkeypatch, reject_threshold
):
    backend = _backend(monkeypatch, reject_threshold=reject_threshold)
    case = _make_verify_case(backend)
    slots, params_by_slot, logits, candidates, draft, starts, frontiers = case
    actual = _verify(
        backend,
        slots=slots,
        logits=logits,
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=starts,
        frontiers=frontiers,
    )
    expected = _direct_verify_reference(
        backend,
        slots=slots,
        params_by_slot=params_by_slot,
        logits=logits,
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=starts,
        frontiers=frontiers,
    )
    for got, ref in zip(actual, expected[:3], strict=True):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
    _assert_probabilities(
        expected[3],
        _reference_probs(logits, params_by_slot, [5, 5, 5, 5, 2, 2, 7]),
    )
    assert actual[1].tolist() == [4, 1, 1]
    assert actual[0].view(3, WIDTH)[0, :3].tolist() == candidates[0, 1:].tolist()
    # The fresh slot's sentinel rejects immediately; q_i=1 draws a bonus.
    assert actual[0].view(3, WIDTH)[1, 0].item() in range(VOCAB)
    assert actual[2][1].tolist() == [4, -1, -1, -1]
    assert actual[2][2].tolist() == [8, -1, -1, -1]


def test_dspark_verify_fullwidth_matches_explicit_csr(monkeypatch):
    backend = _backend(monkeypatch)
    slots, _, logits, candidates, draft, starts, frontiers = _make_verify_case(
        backend, starts=[0, WIDTH, 2 * WIDTH, 3 * WIDTH]
    )
    fullwidth = _verify(
        backend,
        slots=slots,
        logits=logits,
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=None,
        frontiers=frontiers,
    )
    fullwidth = tuple(value.clone() for value in fullwidth)
    packed = _verify(
        backend,
        slots=slots,
        logits=logits,
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=starts,
        frontiers=frontiers,
    )
    for got, ref in zip(packed, fullwidth, strict=True):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_dspark_logprobs_match_fullwidth_and_follow_grammar_csr(monkeypatch):
    backend = _backend(monkeypatch, output_logprobs=True)
    slots, _, logits, candidates, draft, starts, frontiers = _make_verify_case(
        backend, starts=[0, WIDTH, 2 * WIDTH, 3 * WIDTH]
    )
    vocab_mask = _grammar_rows(logits.shape[0])
    common = dict(
        slots=slots,
        candidates=candidates,
        draft_probs=draft,
        frontiers=frontiers,
        vocab_mask=vocab_mask,
    )
    fullwidth = _verify(
        backend,
        logits=logits.clone(),
        query_start_loc=None,
        **common,
    )
    fullwidth = tuple(value.clone() for value in fullwidth)
    packed = _verify(
        backend,
        logits=logits.clone(),
        query_start_loc=starts,
        **common,
    )
    for got, reference in zip(packed, fullwidth, strict=True):
        torch.testing.assert_close(got, reference, atol=0, rtol=0)
    _assert_packed_logprobs(
        fullwidth[3], logits, vocab_mask, fullwidth[0], fullwidth[2], starts
    )

    # Shorten to four, two and one rows. The flat acceptance indices still use
    # request*WIDTH+column, while their target rows follow this different CSR.
    short_starts = torch.tensor([0, 4, 6, 7], device="cuda", dtype=torch.int64)
    selected = torch.tensor([0, 1, 2, 3, 4, 5, 8], device="cuda")
    short_logits = logits.index_select(0, selected)
    short_mask = vocab_mask.index_select(0, selected)
    shortened = _verify(
        backend,
        slots=slots,
        logits=short_logits.clone(),
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=short_starts,
        frontiers=frontiers,
        vocab_mask=short_mask,
    )
    _assert_packed_logprobs(
        shortened[3],
        short_logits,
        short_mask,
        shortened[0],
        shortened[2],
        short_starts,
    )


def test_dspark_prefill_sample_applies_grammar_and_reports_logprobs(monkeypatch):
    backend = _backend(monkeypatch, max_bs=3, output_logprobs=True)
    params = [
        _params(3101, 0.8, 8, 0.9),
        _params(3202, 1.1, 8, 0.9),
        _params(3303, 0.9, 8, 0.9),
    ]
    slots = _prepare(backend, [201, 202, 203], [5, 2, 7], params)
    frontiers = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    frontiers[slots] = torch.tensor([30, 50, 90], device="cuda", dtype=torch.int32)
    logits = _logits(3)
    allowed = torch.tensor([[3], [17], [VOCAB - 1]], device="cuda")
    vocab_mask = _allow_tokens(allowed)
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=frontiers,
        vocab_size=VOCAB,
        vocab_mask=vocab_mask,
        apply_vocab_mask=_vocab_mask_applier(),
        device="cuda",
    )
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    sampled, lengths = backend.sample(output, info)
    assert sampled.tolist() == allowed.squeeze(1).tolist()
    assert lengths.tolist() == [1, 1, 1]
    reference = torch.log_softmax(
        _masked_reference(logits, vocab_mask), dim=-1, dtype=torch.float32
    )
    expected = reference.gather(1, sampled.long().unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(
        output.next_token_logprobs, expected, atol=1e-6, rtol=1e-6
    )


def test_dspark_verify_is_stable_under_request_reordering(monkeypatch):
    backend = _backend(monkeypatch)
    case = _make_verify_case(backend)
    slots, params_by_slot, logits, candidates, draft, starts, frontiers = case
    actual = _verify(
        backend,
        slots=slots,
        logits=logits,
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=starts,
        frontiers=frontiers,
    )
    actual = tuple(value.clone() for value in actual)

    permutation = torch.tensor([2, 0, 1], device="cuda")
    counts = starts[1:] - starts[:-1]
    reordered_counts = counts[permutation]
    reordered_starts = torch.cat(
        [torch.zeros(1, device="cuda", dtype=torch.int64), reordered_counts.cumsum(0)]
    )
    segments = [logits[starts[row] : starts[row + 1]] for row in permutation.tolist()]
    reordered_logits = torch.cat(segments)
    reordered_case = (
        slots[permutation].contiguous(),
        {slot: params_by_slot[slot] for slot in slots[permutation].tolist()},
        reordered_logits,
        candidates[permutation].contiguous(),
        draft,
        reordered_starts,
        frontiers,
    )
    (
        r_slots,
        r_params,
        r_logits,
        r_candidates,
        r_draft,
        r_starts,
        r_frontiers,
    ) = reordered_case
    reordered = _verify(
        backend,
        slots=r_slots,
        logits=r_logits,
        candidates=r_candidates,
        draft_probs=r_draft,
        query_start_loc=r_starts,
        frontiers=r_frontiers,
    )
    reordered_reference = _direct_verify_reference(
        backend,
        slots=r_slots,
        params_by_slot=r_params,
        logits=r_logits,
        candidates=r_candidates,
        draft_probs=r_draft,
        query_start_loc=r_starts,
        frontiers=r_frontiers,
    )
    for got, ref in zip(reordered, reordered_reference[:3], strict=True):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        reordered[0].view(3, WIDTH)[inverse], actual[0].view(3, WIDTH), atol=0, rtol=0
    )
    torch.testing.assert_close(reordered[1][inverse], actual[1], atol=0, rtol=0)
    width = candidates.shape[1]
    offsets = torch.arange(3, device="cuda", dtype=torch.int32)[:, None] * width
    reordered_indices = torch.where(reordered[2] >= 0, reordered[2] - offsets, -1)
    actual_indices = torch.where(actual[2] >= 0, actual[2] - offsets, -1)
    torch.testing.assert_close(
        reordered_indices[inverse], actual_indices, atol=0, rtol=0
    )


def test_dspark_verify_graph_replay_reads_new_csr_and_positions(monkeypatch):
    backend = _backend(monkeypatch)
    case = _make_verify_case(backend)
    slots, params_by_slot, logits, candidates, draft, starts, frontiers = case
    _verify(
        backend,
        slots=slots,
        logits=logits,
        candidates=candidates,
        draft_probs=draft,
        query_start_loc=starts,
        frontiers=frontiers,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = _verify(
            backend,
            slots=slots,
            logits=logits,
            candidates=candidates,
            draft_probs=draft,
            query_start_loc=starts,
            frontiers=frontiers,
        )
    for new_counts in ([2, 4, 1], [1, 2, 4], [4, 1, 2]):
        starts.copy_(
            torch.tensor(
                [0, *torch.tensor(new_counts).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int64,
            )
        )
        frontiers[slots] += 37
        graph.replay()
        captured = (
            graph_result[0].clone(),
            graph_result[1].clone(),
            graph_result[2].clone(),
        )
        eager_result = _verify(
            backend,
            slots=slots,
            logits=logits,
            candidates=candidates,
            draft_probs=draft,
            query_start_loc=starts,
            frontiers=frontiers,
        )
        for got, ref in zip(captured, eager_result, strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
        direct = _direct_verify_reference(
            backend,
            slots=slots,
            params_by_slot=params_by_slot,
            logits=logits,
            candidates=candidates,
            draft_probs=draft,
            query_start_loc=starts,
            frontiers=frontiers,
        )
        for got, ref in zip(eager_result, direct[:3], strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
