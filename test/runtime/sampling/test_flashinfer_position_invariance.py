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

"""Real CUDA sampling acceptance tests; missing CUDA/dependencies fail.

Constructed logits isolate sampling from model numerics. No numerical kernel
or communication is mocked. Multi-token FlashInfer verification requires NVIDIA
CUDA.
"""

import sys
from pathlib import Path

import pytest
import torch

# CI discovers this registration without importing GPU dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.sampling.sampling_batch_info import (
    SamplingBatchInfo,
)

VOCAB = 512
POOL = 8
CHAIN_WIDTH = 4

requires_flashinfer_chain = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="FlashInfer chain verification requires an NVIDIA CUDA GPU",
)


@pytest.fixture(params=["flashinfer", "flashinfer_full"])
def make_backend(request):
    assert torch.cuda.is_available(), "This acceptance suite requires a real CUDA GPU"
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.flashinfer_full import (
        FlashInferFullSamplingBackend,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    previous = global_server_args_dict.get("numerics")
    global_server_args_dict["numerics"] = "rl-bitwise"
    cls = (
        FlashInferSamplingBackend
        if request.param == "flashinfer"
        else FlashInferFullSamplingBackend
    )

    def create(*, width, recorded_proposals):
        return cls(
            SamplingBackendConfig(
                synthetic_acceptance_length=None,
                enable_speculative_sampling=recorded_proposals,
                sampling_stream="per-request",
                logprob_order="torch",
                enable_output_logprobs=True,
                max_bs=4,
                max_draft_tokens_per_req=width,
                max_req_pool_size=POOL,
                vocab_size=VOCAB,
                device="cuda",
            )
        )

    try:
        yield create
    finally:
        if previous is None:
            global_server_args_dict.pop("numerics", None)
        else:
            global_server_args_dict["numerics"] = previous


@pytest.fixture
def backend(make_backend):
    return make_backend(width=1, recorded_proposals=False)


def prepare(backend, slots, *, seed, top_k):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    params = []
    rids = [f"seed-{seed}-k-{top_k}-slot-{slot}" for slot in slots]
    for rid in rids:
        sp = SamplingParams(temperature=0.8, top_k=top_k, top_p=0.95, seed=seed)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)
    backend.prepare_step(
        [(seed * (VOCAB + 2) + top_k + 1) * POOL + slot + 1 for slot in slots],
        slots,
        params,
        num_tokens_per_req=1,
    )


def draw(backend, logits, info, *, decode):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    if decode:
        # Candidate content must not influence an N=1 draw.
        candidates = torch.full(
            (logits.shape[0], 1), 17, device="cuda", dtype=torch.int32
        )
        tokens, lengths = backend.verify(output, info, candidates, tree=None)
    else:
        tokens, lengths = backend.sample(output, info)
    return tokens.clone(), lengths.clone(), output.next_token_logprobs.clone()


def assert_exact(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == e.dtype and a.shape == e.shape
        # Compare logprob bits as well as integer outputs (including signed zero).
        assert torch.equal(a.view(torch.int32), e.view(torch.int32))


@pytest.mark.parametrize("top_k", [1, 50, -1])
def test_prefill_decode_and_chunk_count_invariance(backend, top_k):
    generator = torch.Generator(device="cuda").manual_seed(205)
    logits = torch.randn(1, VOCAB, generator=generator, device="cuda") * 0.3
    slots = torch.tensor([3], device="cuda")
    frontier = torch.zeros(POOL + 1, device="cuda", dtype=torch.int32)
    info = SamplingBatchInfo(req_pool_indices=slots, valid_cache_lengths=frontier)
    info.extend_lengths = torch.empty(1, dtype=torch.int32, device="cuda")
    seed_results = []
    for seed in (45, 46):
        results = []
        for position in range(2049, 2065):
            reference = None
            # Actual cold/warm chunk counts and a one-chunk recomputation.
            for prefix in (0, 130, 514, 1026):
                frontier[3] = prefix
                while prefix < position:
                    chunk = min(1024, position - prefix)
                    prepare(backend, [3], seed=seed, top_k=top_k)
                    info.extend_lengths.fill_(chunk)
                    prefix += chunk
                    if prefix < position:
                        frontier[3] = prefix
                # The frontier still precedes the last chunk until its sample.
                actual = draw(backend, logits, info, decode=False)
                if reference is None:
                    reference = actual
                else:
                    assert_exact(actual, reference)
                frontier[3] = position - 1
                prepare(backend, [3], seed=seed, top_k=top_k)
                decode_info = SamplingBatchInfo(
                    req_pool_indices=slots, valid_cache_lengths=frontier
                )
                assert_exact(draw(backend, logits, decode_info, decode=True), reference)
            results.append(reference[0])
        seed_results.append(torch.cat(results))
    if top_k != 1:
        assert not torch.equal(*seed_results), "Different seeds must exercise sampling"
        assert (
            torch.unique(seed_results[0]).numel() > 1
        ), "Positions must change the draw"


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "cuda_graph"])
def test_batch_slot_and_replay_invariance(backend, use_graph):
    generator = torch.Generator(device="cuda").manual_seed(207)
    target_logits = torch.randn(1, VOCAB, generator=generator, device="cuda") * 0.3
    frontier = torch.zeros(POOL + 1, device="cuda", dtype=torch.int32)
    reference_req = torch.tensor([7], device="cuda")
    reference_info = SamplingBatchInfo(
        req_pool_indices=reference_req, valid_cache_lengths=frontier
    )
    # Reference computed in a different slot, by prefill, before any capture.
    reference = []
    for position in range(2049, 2065):
        prepare(backend, [7], seed=45, top_k=50)
        frontier[7] = 130
        reference_info.extend_lengths = torch.tensor(
            [position - 130], device="cuda", dtype=torch.int32
        )
        reference.append(draw(backend, target_logits, reference_info, decode=False))

    for width in (1, 4):
        req = torch.arange(1, width + 1, device="cuda", dtype=torch.int64)
        logits = torch.randn(width, VOCAB, generator=generator, device="cuda")
        for slot in range(1, width + 1):
            prepare(backend, [slot], seed=45, top_k=50)

        # Bind once before warmup/capture. Replay updates only tensor contents.
        info = SamplingBatchInfo(req_pool_indices=req, valid_cache_lengths=frontier)
        position_address = backend._sampling_offsets_pool.data_ptr()

        def forward():
            return draw(backend, logits, info, decode=True)

        if use_graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()

        for step, position in enumerate(range(2049, 2065)):
            # Change both the row and pool mapping without recapture.
            row = step % width
            order = list(range(1, width + 1))
            if step % 2:
                order.reverse()
            req.copy_(torch.tensor(order, device="cuda"))
            frontier.fill_(position - 1)
            logits.copy_(torch.randn(width, VOCAB, generator=generator, device="cuda"))
            logits[row].copy_(target_logits[0])
            prepare(backend, order, seed=45, top_k=50)
            if use_graph:
                graph.replay()
                actual = captured
            else:
                actual = forward()
            assert backend._sampling_offsets_pool.data_ptr() == position_address
            assert_exact(tuple(x[row : row + 1] for x in actual), reference[step])


def test_mixed_prefill_and_decode_share_positions(backend):
    generator = torch.Generator(device="cuda").manual_seed(209)
    target = torch.randn(1, VOCAB, generator=generator, device="cuda") * 0.3
    slots = torch.tensor([2, 4, 1, 3], device="cuda")
    prepare(backend, slots.tolist(), seed=45, top_k=50)
    frontier = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    frontier[slots] = torch.tensor(
        [130, 514, 2048, 2048], dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor([1919, 1535, 1, 1], dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(req_pool_indices=slots, valid_cache_lengths=frontier)
    info.extend_lengths = lengths[:2]
    prefill_info, decode_info = info[:2], info[2:]
    # Same request-level slicing as the executor: full position pool survives,
    # and the decode suffix must not read a batch-row-indexed coin buffer.
    prefill = draw(
        backend, target.expand(2, -1).contiguous(), prefill_info, decode=False
    )
    decode = draw(backend, target.expand(2, -1).contiguous(), decode_info, decode=True)
    assert_exact(decode, prefill)


def test_single_verify_publishes_one_token_and_updates_counts_once(backend):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.flashinfer_full import (
        FlashInferFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    full = isinstance(backend, FlashInferFullSamplingBackend)
    slots = torch.tensor([1, 3], device="cuda")
    params = []
    for rid in ("pack-a", "pack-b"):
        sp = SamplingParams(
            temperature=0.8,
            top_k=1,
            top_p=0.95,
            seed=45,
            frequency_penalty=2.0 if full else 0.0,
        )
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)
    backend.prepare_step([26, 27], [1, 3], params, num_tokens_per_req=1)
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=torch.full(
            (POOL + 1,), 2048, device="cuda", dtype=torch.int32
        ),
    )
    logits = torch.zeros(2, VOCAB, device="cuda")
    logits[0, 5] = logits[0, 7] = 10  # Canonical greedy tie: the smaller id wins.
    logits[1, 17], logits[1, 19] = 10, 9
    candidates = torch.zeros(2, 1, dtype=torch.int32, device="cuda")
    addresses = None
    for step in range(2):
        tokens, lengths = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits.clone()),
            info,
            candidates,
            tree=None,
        )
        expected = [7, 19] if full and step else [5, 17]
        assert tokens.tolist() == expected
        assert lengths.tolist() == [1, 1]
        assert backend.accepted_path(2, 1).tolist() == [[0], [1]]
        current_addresses = (tokens.data_ptr(), lengths.data_ptr())
        if addresses is not None:
            assert current_addresses == addresses
        addresses = current_addresses
        packed = backend.get_packed_output_d2h(tokens, lengths)
        assert packed is not None
        torch.cuda.synchronize()
        assert packed[0].tolist() == expected
        assert packed[1].tolist() == [1, 1]
        if full:
            counts = backend._counts.index_select(0, slots)
            assert counts.sum(dim=1).tolist() == [step + 1, step + 1]
            assert counts[:, [5, 7, 17, 19]].tolist() == (
                [[1, 0, 0, 0], [0, 0, 1, 0]]
                if step == 0
                else [[1, 1, 0, 0], [0, 0, 1, 1]]
            )


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "cuda_graph"])
def test_repeated_padding_does_not_change_a_real_request(backend, use_graph):
    generator = torch.Generator(device="cuda").manual_seed(211)
    logits = torch.randn(4, VOCAB, generator=generator, device="cuda")
    slots = torch.tensor([3, 0, 0, 0], dtype=torch.int64, device="cuda")
    frontier = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    frontier[3] = 2048
    prepare(backend, [3], seed=45, top_k=50)
    solo_info = SamplingBatchInfo(
        req_pool_indices=slots[:1], valid_cache_lengths=frontier
    )
    expected = draw(backend, logits[:1], solo_info, decode=True)
    info = SamplingBatchInfo(req_pool_indices=slots, valid_cache_lengths=frontier)
    if use_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            draw(backend, logits, info, decode=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = draw(backend, logits, info, decode=True)
        graph.replay()
    else:
        actual = draw(backend, logits, info, decode=True)
    assert_exact(tuple(t[:1] for t in actual), expected)
    assert backend._sampling_offsets_pool[0].item() == 0


def test_prefill_graph_reads_current_frontier_and_lengths(backend):
    generator = torch.Generator(device="cuda").manual_seed(213)
    logits = torch.randn(1, VOCAB, generator=generator, device="cuda") * 0.3
    slots = torch.tensor([3], dtype=torch.int64, device="cuda")
    frontier = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2049], dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(
        req_pool_indices=slots, valid_cache_lengths=frontier, extend_lengths=lengths
    )
    prepare(backend, [3], seed=45, top_k=50)
    expected = draw(backend, logits, info, decode=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        draw(backend, logits, info, decode=False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = draw(backend, logits, info, decode=False)
    for prefix in (0, 130, 514, 1026, 2048):
        frontier[3] = prefix
        lengths.fill_(2049 - prefix)
        graph.replay()
        assert_exact(actual, expected)
        assert frontier[3].item() == prefix


def _chain_inputs(recorded_proposals):
    logits = torch.full((CHAIN_WIDTH, VOCAB), -torch.inf, device="cuda")
    logits[:, 1:4] = torch.tensor([0.25, 0.50, 0.25], device="cuda").log()
    # Column zero is the previous verified token; the first proposal has
    # p=0.25 and, for recorded proposals, q=0.50.
    candidates = torch.tensor([[4, 1, 1, 1]], device="cuda", dtype=torch.int32)
    recorded = None
    if recorded_proposals:
        recorded = torch.zeros(POOL + 1, CHAIN_WIDTH, VOCAB, device="cuda")
        recorded[:, :-1, 1:4] = torch.tensor([0.50, 0.375, 0.125], device="cuda")
    return logits, candidates, recorded


def _chain_params(seed):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    params = SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=seed)
    params.resolve_seed(f"chain-{seed}")
    params.normalize(None)
    return params


def _verify_chain(backend, logits, info, candidates):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    output = LogitsProcessorOutput(next_token_logits=logits)
    tokens, lengths = backend.verify(output, info, candidates, tree=None)
    bs, width = candidates.shape
    return (
        tokens.view(bs, width).clone(),
        lengths.clone(),
        output.next_token_logprobs.view(bs, width).clone(),
        backend.accepted_path(bs, width).clone(),
    )


def _emitted_chain_row(result, row):
    tokens, lengths, logprobs, paths = result
    length = lengths[row : row + 1]
    kept = slice(None, length.item())
    return (
        tokens[row, kept],
        length,
        logprobs[row, kept],
        paths[row, kept] - row * tokens.shape[1],
    )


@requires_flashinfer_chain
@pytest.mark.parametrize("recorded_proposals", [False, True], ids=["target_only", "q"])
def test_chain_verify_ignores_extra_prefill_prepare(make_backend, recorded_proposals):
    backend = make_backend(width=CHAIN_WIDTH, recorded_proposals=recorded_proposals)
    logits, candidates, recorded = _chain_inputs(recorded_proposals)
    params = _chain_params(seed=45)
    # Non-emitting prefill still passes the deployment's chain width to
    # prepare_step. With generator refills, seed 45 moves the first accept
    # coin across both the target-only and recorded-q thresholds.
    backend.prepare_step([1], [1], [params], num_tokens_per_req=CHAIN_WIDTH)
    backend.prepare_step(
        [3, 1, 2],
        [3, 1, 2],
        [_chain_params(seed=46), params, params],
        num_tokens_per_req=CHAIN_WIDTH,
    )
    mixed_info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([3, 1, 2], device="cuda"),
        valid_cache_lengths=torch.full(
            (POOL + 1,), 2048, dtype=torch.int32, device="cuda"
        ),
        extend_lengths=torch.tensor([37], device="cuda", dtype=torch.int32),
        draft_probs=recorded,
    )
    actual = _verify_chain(
        backend, logits.repeat(2, 1), mixed_info[1:], candidates.repeat(2, 1)
    )
    assert_exact(_emitted_chain_row(actual, 0), _emitted_chain_row(actual, 1))


@requires_flashinfer_chain
def test_chain_verify_resumes_from_sampler_snapshot(make_backend):
    backend = make_backend(width=CHAIN_WIDTH, recorded_proposals=True)
    logits, candidates, recorded = _chain_inputs(recorded_proposals=True)
    params = _chain_params(seed=45)
    backend.prepare_step(
        [1, 2], [1, 2], [params, _chain_params(seed=79)], num_tokens_per_req=CHAIN_WIDTH
    )
    image = torch.empty(backend.slot_state_bytes(), dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.current_stream()
    backend.export_slot_state(1, image, stream, admission_id=1)
    stream.synchronize()
    backend.import_slot_state(2, image, stream, admission_id=1)
    backend.prepare_step([1], [2], [params], num_tokens_per_req=CHAIN_WIDTH)

    # RuntimeStates supplies the restored frontier and recorded proposals.
    frontier = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([1, 2], device="cuda"),
        valid_cache_lengths=frontier,
        draft_probs=recorded,
    )
    logits = logits.repeat(2, 1)
    candidates = candidates.repeat(2, 1)
    for position in range(2048, 2064):
        frontier[1:3].fill_(position)
        actual = _verify_chain(backend, logits, info, candidates)
        assert_exact(_emitted_chain_row(actual, 0), _emitted_chain_row(actual, 1))


@requires_flashinfer_chain
@pytest.mark.parametrize("recorded_proposals", [False, True], ids=["target_only", "q"])
def test_chain_verify_replay_reads_current_seed_and_position(
    make_backend, recorded_proposals
):
    backend = make_backend(width=CHAIN_WIDTH, recorded_proposals=recorded_proposals)
    logits, candidates, recorded = _chain_inputs(recorded_proposals)
    frontier = torch.arange(POOL + 1, dtype=torch.int32, device="cuda")
    solo = SamplingBatchInfo(
        req_pool_indices=torch.tensor([7], device="cuda"),
        valid_cache_lengths=frontier,
        draft_probs=recorded,
    )
    slots = torch.tensor([3, 4, 0, 0], device="cuda")
    packed = SamplingBatchInfo(
        req_pool_indices=slots, valid_cache_lengths=frontier, draft_probs=recorded
    )
    packed_logits = logits.repeat(4, 1).view(4, CHAIN_WIDTH, VOCAB)
    packed_candidates = candidates.repeat(4, 1)
    other_logits = logits.roll(1, dims=-1)
    other_q = torch.tensor([0.875, 0.0625, 0.0625], device="cuda")
    other_params = _chain_params(seed=79)
    backend.prepare_step(
        [3, 4], [3, 4], [other_params, other_params], num_tokens_per_req=CHAIN_WIDTH
    )

    def forward():
        return _verify_chain(
            backend, packed_logits.flatten(0, 1), packed, packed_candidates
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()

    seed_results = []
    for seed in (45, 46):
        params = _chain_params(seed)
        results = []
        for step, position in enumerate(range(2049, 2081)):
            backend.prepare_step([seed], [7], [params], num_tokens_per_req=CHAIN_WIDTH)
            frontier[7] = position - 1
            expected = _emitted_chain_row(
                _verify_chain(backend, logits, solo, candidates), 0
            )
            row = step % 2
            probe_slot = 3 + row
            order = [3, 4] if step % 4 < 2 else [4, 3]
            probe_row = order.index(probe_slot)
            other_slot = order[1 - probe_row]
            packed_logits[probe_row].copy_(logits)
            packed_logits[1 - probe_row].copy_(other_logits)
            slots[:2].copy_(torch.tensor(order, device="cuda"))
            frontier[probe_slot] = position - 1
            frontier[other_slot] = position + 123
            if recorded is not None:
                recorded[probe_slot].copy_(recorded[7])
                recorded[other_slot, :-1, 1:4] = other_q
            packed_params = [other_params, other_params]
            packed_params[probe_row] = params
            admission_ids = order.copy()
            admission_ids[probe_row] = seed
            backend.prepare_step(
                admission_ids, order, packed_params, num_tokens_per_req=CHAIN_WIDTH
            )
            graph.replay()
            assert_exact(_emitted_chain_row(captured, probe_row), expected)
            results.append(tuple(expected[0].tolist()))
        assert len(set(results)) > 1, "Changing positions must change sampled results"
        assert any(len(tokens) > 1 for tokens in results), "Must accept a proposal"
        seed_results.append(results)
    assert (
        seed_results[0] != seed_results[1]
    ), "Changing seeds must change sampled results"
