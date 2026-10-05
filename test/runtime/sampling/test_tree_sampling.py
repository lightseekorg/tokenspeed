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

"""Sampled draft-tree verify on the triton backend.

Each tree node's target pick is a Gumbel sample keyed by (seed, vc + depth),
the key plain decoding uses at that position, so a node samples exactly what
the chain (and non-speculative decode) would sample there.
"""

import pytest
import torch

from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.tree_verify import TreeVerifyBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB, POOL, BS, N = 512, 8, 3, 4
PARAMS = {
    "greedy": dict(temperature=0.0, top_k=-1, top_p=1.0),
    "top_k_top_p": dict(temperature=0.8, top_k=40, top_p=0.95),
    "top_p": dict(temperature=0.7, top_k=-1, top_p=0.9),
    "temperature": dict(temperature=1.0, top_k=-1, top_p=1.0),
}


def _config(logprobs: bool) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        max_bs=BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
        enable_output_logprobs=logprobs,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
    )


def _backend(
    route: str, logprobs: bool = False
) -> tuple[TritonSamplingBackend, torch.Tensor]:
    backend = TritonSamplingBackend(_config(logprobs))
    params = []
    for i in range(BS):
        sp = SamplingParams(**PARAMS[route])
        sp.resolve_seed(f"tree_{i}")
        sp.normalize(None)
        params.append(sp)
    pool = [2, 5, 7]
    backend.prepare_step(
        request_ids=[f"tree_{i}" for i in range(BS)],
        request_pool_indices=pool,
        sampling_params_list=params,
        num_tokens_per_req=N,
    )
    return backend, torch.tensor(pool, dtype=torch.int64, device="cuda")


def _info(req, offsets) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=req,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
    )


def _tree(parent: list[int], req: torch.Tensor) -> TreeSpec:
    spec = TreeSpec(
        TreeSpecConfig(topk=2, num_steps=N - 1, num_nodes=N),
        BS,
        torch.device("cuda"),
    )
    parent_map = torch.tensor(parent, dtype=torch.int32, device="cuda").repeat(
        POOL + 1, 1
    )
    spec.load_step(BS, req, parent_map)
    return spec


def _batch(tree: TreeSpec) -> TreeVerifyBatch:
    return TreeVerifyBatch(parents=tree.parent_buf[:BS], depths=tree.depth_buf[:BS])


@pytest.mark.parametrize("route", list(PARAMS))
def test_chain_shaped_tree_verifies_like_chain(route):
    torch.manual_seed(0)
    offsets = torch.arange(50, 50 + POOL + 1, dtype=torch.int32, device="cuda")
    logits = torch.randn(BS * N, VOCAB, device="cuda") * 3

    backend, req = _backend(route)
    probe = backend._sample_verify_targets(
        logits,
        req.int(),
        (
            backend._temperature_pool,
            backend._top_k_pool,
            backend._top_p_pool,
            backend._seed_pool,
        ),
        offsets,
        BS * N,
        N,
    ).view(BS, N)
    # Candidates follow the target for two steps, then miss, so the path has depth 2.
    candidates = torch.randint(0, VOCAB, (BS, N), dtype=torch.int32, device="cuda")
    candidates[:, 1:3] = probe[:, 0:2]
    candidates[:, 3] = (probe[:, 2] + 1) % VOCAB

    chain_predict, chain_accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info(req, offsets),
        candidates,
        tree=None,
    )
    chain_predict, chain_accept = chain_predict.clone(), chain_accept.clone()
    tree = _tree([-1, 0, 1, 2], req)
    tree_predict, tree_accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info(req, offsets),
        candidates,
        tree=_batch(tree),
    )
    assert torch.equal(chain_accept, tree_accept)
    assert torch.all(tree_accept == 3)
    for b in range(BS):
        n = int(tree_accept[b])
        assert torch.equal(
            chain_predict[b * N : b * N + n], tree_predict[b * N : b * N + n]
        )
    assert torch.equal(
        backend.accepted_path(BS, N)[:, :3].cpu(),
        torch.tensor([[0, 1, 2]] * BS, dtype=torch.int32),
    )


@pytest.mark.parametrize("route", list(PARAMS))
def test_nodes_sample_the_chain_draw_at_their_depth(route):
    torch.manual_seed(1)
    offsets = torch.arange(70, 70 + POOL + 1, dtype=torch.int32, device="cuda")
    by_depth = torch.randn(BS, N, VOCAB, device="cuda") * 3
    backend, req = _backend(route)
    chain = (
        backend._sample_verify_targets(
            by_depth.view(BS * N, VOCAB),
            req.int(),
            (
                backend._temperature_pool,
                backend._top_k_pool,
                backend._top_p_pool,
                backend._seed_pool,
            ),
            offsets,
            BS * N,
            N,
        )
        .view(BS, N)
        .clone()
    )

    # Siblings 1 and 2 share depth 1; node 3 hangs under node 2 at depth 2.
    tree = _tree([-1, 0, 0, 2], req)
    depth = tree.depth_buf[:BS].long()
    logits = torch.gather(by_depth, 1, depth[:, :, None].expand(-1, -1, VOCAB)).view(
        BS * N, VOCAB
    )
    picks = backend._sample_tree_targets(
        logits, req.int(), offsets, _batch(tree), BS
    ).view(BS, N)
    assert torch.equal(picks, torch.gather(chain, 1, depth))


@pytest.mark.parametrize("backend_name", ["greedy", "triton"])
def test_greedy_tree_verify_walks_the_argmax_with_logprobs(backend_name):
    torch.manual_seed(2)
    offsets = torch.arange(30, 30 + POOL + 1, dtype=torch.int32, device="cuda")
    logits = torch.randn(BS * N, VOCAB, device="cuda") * 3
    target = logits.argmax(-1).view(BS, N)
    if backend_name == "greedy":
        backend = GreedySamplingBackend(_config(True))
        req = torch.tensor([2, 5, 7], dtype=torch.int64, device="cuda")
    else:
        backend, req = _backend("greedy", logprobs=True)

    # Node 1 misses, node 2 (its sibling) matches, node 3 under node 2 matches for request 0 only.
    parent = [-1, 0, 0, 2]
    tree = _tree(parent, req)
    candidates = torch.randint(0, VOCAB, (BS, N), dtype=torch.int32, device="cuda")
    candidates[:, 1] = (target[:, 0] + 1) % VOCAB
    candidates[:, 2] = target[:, 0]
    candidates[:, 3] = (target[:, 2] + 1) % VOCAB
    candidates[0, 3] = target[0, 2]

    out = LogitsProcessorOutput(next_token_logits=logits.clone())
    predict, accept = backend.verify(
        out, _info(req, offsets), candidates, tree=_batch(tree)
    )
    path = backend.accepted_path(BS, N).cpu().tolist()

    logprobs = torch.log_softmax(logits.float(), -1)
    for b in range(BS):
        want_path, node = [0], 0
        while True:
            kids = [
                j
                for j in range(N)
                if parent[j] == node and int(candidates[b, j]) == int(target[b, node])
            ]
            if not kids:
                break
            node = kids[0]
            want_path.append(node)
        assert int(accept[b]) == len(want_path)
        assert path[b] == want_path + [-1] * (N - len(want_path))
        for d, node in enumerate(want_path):
            assert int(predict[b * N + d]) == int(target[b, node])
            torch.testing.assert_close(
                out.next_token_logprobs[b * N + d].float(),
                logprobs[b * N + node, target[b, node]],
                atol=1e-3,
                rtol=1e-3,
            )
