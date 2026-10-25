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
