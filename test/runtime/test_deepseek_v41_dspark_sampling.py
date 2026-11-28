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

"""DSpark samples directly into final proposal views with anchor columns."""

import pytest
import torch

from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.deepseek_v41_dspark_heads import (
    DSparkVanillaMarkov,
    dspark_greedy_workspace,
    sample_dspark_block_greedy,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _inputs(rows, block):
    embedding = VocabParallelEmbedding(101, 32, params_dtype=torch.bfloat16).to("cuda")
    projection = ParallelLMHead(101, 32, params_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        embedding.weight.normal_()
        projection.weight.normal_()
    markov = DSparkVanillaMarkov(embedding, projection)
    logits = torch.randn(
        rows, block, projection.num_embeddings_per_partition, device="cuda"
    )
    candidates, partials = dspark_greedy_workspace(
        1, rows, logits.shape[-1], logits.device
    )
    tokens = torch.full((rows + 4, block + 1), -313, dtype=torch.int32, device="cuda")
    tokens[2 : 2 + rows, 0] = torch.arange(rows, device="cuda", dtype=torch.int32) % 101
    return markov, projection, logits, candidates, partials, tokens


def _sample(markov, head, logits, candidates, partials, anchors, output):
    return sample_dspark_block_greedy(
        logits,
        anchors,
        markov,
        head,
        None,
        candidates,
        partials,
        output,
    )


@pytest.mark.parametrize("rows", [0, 6, 12, 24, 48, 96, 192])
@pytest.mark.parametrize("block", [1, 6])
def test_samples_strided_output_without_touching_anchors_or_other_requests(rows, block):
    markov, head, logits, candidates, partials, tokens = _inputs(rows, block)
    anchors = tokens[2 : 2 + rows, 0]
    before = tokens.clone()
    contiguous = torch.empty(rows, block, dtype=torch.int32, device="cuda")
    _sample(markov, head, logits, candidates, partials, anchors.clone(), contiguous)
    output = tokens[2 : 2 + rows, 1:]
    returned = _sample(markov, head, logits, candidates, partials, anchors, output)
    assert returned is output
    expected = before.clone()
    expected[2 : 2 + rows, 1:] = contiguous
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)


def test_direct_proposals_graph_replay_then_eager():
    rows, block = 24, 6
    markov, head, logits, candidates, partials, tokens = _inputs(rows, block)
    anchors = tokens[2 : 2 + rows, 0]
    output = tokens[2 : 2 + rows, 1:]
    _sample(markov, head, logits, candidates, partials, anchors, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _sample(markov, head, logits, candidates, partials, anchors, output)
    for _ in range(3):
        logits.normal_()
        anchors.copy_((anchors + 7) % 101)
        before = tokens.clone()
        graph.replay()
        eager = torch.empty_like(output, memory_format=torch.contiguous_format)
        _sample(markov, head, logits, candidates, partials, anchors.clone(), eager)
        expected = before.clone()
        expected[2 : 2 + rows, 1:] = eager
        torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [64, 129280])
@pytest.mark.parametrize("block", [1, 5])
def test_fused_sampled_markov_matches_reference_with_filters_and_graph_replay(
    monkeypatch, vocab: int, block: int
):
    from tokenspeed_kernel.ops.sampling.packed_verify import DRAFT_POS_HASH_MUL
    from tokenspeed_kernel.platform import current_platform

    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams
    from tokenspeed.runtime.utils.env import global_server_args_dict

    if not current_platform().is_nvidia:
        pytest.skip("FlashInferSamplingBackend sampled DSpark requires NVIDIA CUDA")
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", True)
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", "standard"
    )
    from tokenspeed.runtime.models.deepseek_v41_dspark_heads import (
        dspark_sampled_workspace,
        sample_dspark_block_sampled,
    )

    rows, rank, pool_rows = 4, 64, 10
    gen = torch.Generator(device="cuda").manual_seed(53)
    embedding = VocabParallelEmbedding(vocab, rank, params_dtype=torch.bfloat16).to(
        "cuda"
    )
    projection = ParallelLMHead(vocab, rank, params_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        embedding.weight.copy_(
            (
                torch.randint(
                    -3, 4, (vocab, rank), device="cuda", generator=gen
                ).float()
                / 8.0
            ).to(torch.bfloat16)
        )
        projection.weight.copy_(
            (
                torch.randint(
                    -3, 4, projection.weight.shape, device="cuda", generator=gen
                ).float()
                / 8.0
            ).to(torch.bfloat16)
        )
    markov = DSparkVanillaMarkov(embedding, projection)
    backend = FlashInferSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="per-request",
            logprob_order="torch",
            max_bs=rows,
            max_draft_tokens_per_req=block + 1,
            max_req_pool_size=pool_rows - 1,
            vocab_size=vocab,
            device="cuda",
        )
    )
    params = [
        SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, seed=10),
        SamplingParams(temperature=1.0, top_k=-1, top_p=0.95, seed=21),
        SamplingParams(temperature=0.8, top_k=min(12, vocab), top_p=0.9, seed=32),
        SamplingParams(temperature=1.15, top_k=-1, top_p=1.0, seed=43),
    ]
    for i, param in enumerate(params):
        param.resolve_seed(f"fused-sampled-{i}")
        param.normalize(None)
    slots_list = [2, 5, 7, 9]
    write_slots_list = [1, 4, 6, 8]
    backend.prepare_step(
        [f"fused-sampled-{i}" for i in range(rows)],
        slots_list,
        params,
        block + 1,
    )
    slots = torch.tensor(slots_list, device="cuda", dtype=torch.int64)
    write_slots = torch.tensor(write_slots_list, device="cuda", dtype=torch.int64)
    ws = dspark_sampled_workspace(
        max_rows=rows,
        vocab_size=vocab,
        rank=rank,
        embed_dtype=embedding.weight.dtype,
        pools=backend.speculative_sampling_pools(),
        topk_topp_workspace_bytes=backend.dspark_topk_topp_workspace_size(rows, vocab),
        renorm_probs=backend.renorm_dspark_probabilities,
        device="cuda",
    )

    base_logits = torch.randn(
        (rows, block, projection.num_embeddings_per_partition),
        device="cuda",
        generator=gen,
    )
    start_pos = torch.tensor([11, 77, 133, 211], device="cuda", dtype=torch.int64)
    tokens = torch.full((rows + 2, block + 1), -77, device="cuda", dtype=torch.int32)
    tokens[1 : 1 + rows, 0] = (
        torch.arange(rows, device="cuda", dtype=torch.int32) * 5 + 3
    )
    fused_draft_probs = torch.full(
        (pool_rows, block, vocab), -2.0, device="cuda", dtype=torch.float32
    )
    ref_draft_probs = fused_draft_probs.clone()
    ref_tokens = tokens[1 : 1 + rows].clone()

    prev = ref_tokens[:, 0].clamp(0, vocab - 1)
    for step in range(block):
        emb = markov.embedding(prev)
        bias = torch.mm(emb, markov.projection.weight.T, out_dtype=torch.float32)
        logits = (base_logits[:, step] + bias)[:, :vocab].contiguous()
        sampled = torch.empty((rows,), dtype=torch.int32, device="cuda")
        probabilities = backend.sample_dspark_proposals(
            logits,
            slots,
            start_pos + 1 + (step + 1) * DRAFT_POS_HASH_MUL,
            sampled,
        )
        ref_draft_probs[:, step].index_copy_(0, write_slots, probabilities)
        ref_tokens[:, step + 1].copy_(sampled)
        prev = sampled

    anchors = tokens[1 : 1 + rows, 0]
    output = tokens[1 : 1 + rows, 1:]
    before_tokens = tokens.clone()
    returned = sample_dspark_block_sampled(
        base_logits,
        anchors,
        markov,
        slots,
        write_slots,
        start_pos,
        fused_draft_probs,
        ws,
        vocab,
        output,
    )
    assert returned is output
    assert torch.equal(tokens[:, 0], before_tokens[:, 0])
    assert torch.equal(tokens[0], before_tokens[0])
    assert torch.equal(tokens[-1], before_tokens[-1])
    torch.testing.assert_close(
        fused_draft_probs[write_slots],
        ref_draft_probs[write_slots],
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(tokens[1 : 1 + rows, 1:], ref_tokens[:, 1:])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sample_dspark_block_sampled(
            base_logits,
            anchors,
            markov,
            slots,
            write_slots,
            start_pos,
            fused_draft_probs,
            ws,
            vocab,
            output,
        )
    base_logits.normal_()
    start_pos += 29
    anchors.copy_((anchors + 5) % vocab)
    fused_draft_probs.fill_(-5.0)
    graph.replay()
    captured_tokens = captured.clone()
    captured_probs = fused_draft_probs[write_slots].clone()
    fused_draft_probs.fill_(-5.0)
    eager = torch.empty_like(captured_tokens, memory_format=torch.contiguous_format)
    sample_dspark_block_sampled(
        base_logits,
        anchors,
        markov,
        slots,
        write_slots,
        start_pos,
        fused_draft_probs,
        ws,
        vocab,
        eager,
    )
    assert torch.equal(captured_tokens, eager)
    torch.testing.assert_close(
        captured_probs, fused_draft_probs[write_slots], atol=0, rtol=0
    )


def test_small_vocab_topk1_matches_fused_topk_topp_renorm():
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        _FUSED_TOPK_TOPP_AVAILABLE,
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    if not _FUSED_TOPK_TOPP_AVAILABLE:
        pytest.skip("requires fused_topk_topp CUDA extension")

    vocab = 1024
    rows = 3
    pool_rows = 4
    backend = FlashInferSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="per-request",
            logprob_order="torch",
            max_bs=rows,
            max_draft_tokens_per_req=2,
            max_req_pool_size=pool_rows - 1,
            vocab_size=vocab,
            device="cuda",
        )
    )
    params = [
        SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, seed=1),
        SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, seed=2),
        SamplingParams(temperature=1.0, top_k=1, top_p=0.9, seed=3),
    ]
    for i, param in enumerate(params):
        param.resolve_seed(f"topk1-{i}")
        param.normalize(None)
    slots_list = [1, 2, 3]
    backend.prepare_step([f"topk1-{i}" for i in range(rows)], slots_list, params, 2)
    slots = torch.tensor(slots_list, device="cuda", dtype=torch.int64)

    logits = torch.full((rows, vocab), -4.0, device="cuda", dtype=torch.float32)
    logits[0, 17] = 2.5
    logits[1, [9, 255]] = 3.0
    logits[2, [0, 512, 1023]] = 1.5

    fused = backend._dspark_probabilities(logits, slots)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    top_ks = backend._top_k_pool.index_select(0, slots)
    top_ps = backend._top_p_pool.index_select(0, slots)
    small_vocab = backend._apply_small_vocab_topk_topp(
        probabilities,
        top_ks,
        top_ps,
        greedy_one_hot=None,
        logits=logits,
        in_place=False,
    )
    torch.testing.assert_close(small_vocab, fused, atol=0, rtol=0)
    assert torch.equal(
        small_vocab.argmax(dim=-1), torch.tensor([17, 9, 0], device="cuda")
    )
