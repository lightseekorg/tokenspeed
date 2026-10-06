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

"""DraftTree against a plain per-request reference of EAGLE-2 tree drafting."""

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton.draft_tree import (
    draft_tree_expand,
    draft_tree_finalize,
    tree_ancestry,
)
from tokenspeed_kernel.ops.sampling.triton.logprob_topk import logprob_topk

from tokenspeed.runtime.execution.drafter.tree import DraftTree

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = "cuda"


def _drive(bs, topk, steps, nodes, vocab, seed):
    """Run DraftTree on random log-probs; return its output and the entry record."""
    gen = torch.Generator().manual_seed(seed)
    tree = DraftTree(bs, topk, steps, nodes, torch.device(DEVICE))
    record = []  # per request: list of (token path tuple, score)
    lane_paths = [[None] * topk for _ in range(bs)]
    logp = torch.log_softmax(torch.randn(bs, vocab, generator=gen) * 3, -1).to(DEVICE)
    tree.seed(bs, *logprob_topk(logp, topk))
    for b in range(bs):
        sc, tk = torch.topk(logp[b].float(), topk)
        record.append([((int(t),), float(s)) for t, s in zip(tk, sc)])
        lane_paths[b] = [((int(t),), float(s)) for t, s in zip(tk, sc)]
    for step in range(1, steps):
        logp = torch.log_softmax(
            torch.randn(bs * topk, vocab, generator=gen) * 3, -1
        ).to(DEVICE)
        lane_tokens = tree.expand(bs, step, *logprob_topk(logp, topk), None)
        for b in range(bs):
            cands = []
            for lane in range(topk):
                path, score = lane_paths[b][lane]
                sc, tk = torch.topk(logp[b * topk + lane].float(), topk)
                for t, s in zip(tk, sc):
                    cands.append((path + (int(t),), score + float(s)))
            record[b].extend(cands)
            best = sorted(range(len(cands)), key=lambda i: -cands[i][1])[:topk]
            lane_paths[b] = [cands[i] for i in best]
            got = lane_tokens[b].tolist()
            assert got == [cands[i][0][-1] for i in best]
    # A strided column, as the drafter passes it.
    roots = (torch.arange(bs, device=DEVICE, dtype=torch.int32) + 7)[:, None].repeat(
        1, 3
    )[:, 0]
    tokens, parent = tree.finalize(bs, roots)
    return tokens.cpu(), parent.cpu(), record


def test_finalize_keeps_the_best_scores_even_when_nearly_tied():
    """Distinct scores closer than any depth penalty keep their order."""
    scores = torch.tensor([[-1.0, -1.0000003576, -1.0000008345]], device=DEVICE)
    parent = torch.tensor([[-1, 0, -1]], device=DEVICE)
    depth = torch.tensor([[1, 2, 1]], device=DEVICE)
    tokens = torch.tensor([[10, 11, 12]], device=DEVICE)
    root = torch.tensor([9], device=DEVICE)
    out_tokens, out_parent = draft_tree_finalize(
        scores, parent, depth, tokens, root, num_nodes=3, max_depth=2, rank_bits=6
    )
    assert out_tokens.tolist() == [[9, 10, 11]]
    assert out_parent.tolist() == [[-1, 0, 1]]


def test_positive_child_score_cannot_outrank_its_parent():
    """A child scoring above its parent would leave a kept node's parent unkept."""
    dev = torch.device(DEVICE)
    # Two seed lanes: entry 1 (-0.5) and entry 0 (-1.0); each has two children.
    entry_scores = torch.tensor([[-1.0, -0.5, 0, 0, 0, 0]], device=dev)
    entry_parent = torch.tensor([[-1, -1, -1, -1, -1, -1]], device=dev)
    entry_depth = torch.tensor([[1, 1, 0, 0, 0, 0]], device=dev)
    entry_tokens = torch.tensor([[10, 11, 0, 0, 0, 0]], device=dev)
    lane_scores = torch.tensor([[-0.5, -1.0]], device=dev)
    lane_entry = torch.tensor([[1, 0]], device=dev)
    # Lane 1 (entry 0) proposes a child at +0.6: raw, it would outscore entry 0.
    child_scores = torch.tensor([[-3.0, -4.0, 0.6, -5.0]], device=dev)
    child_tokens = torch.tensor([[20, 21, 22, 23]], device=dev)
    draft_tree_expand(
        child_scores,
        child_tokens,
        lane_scores,
        lane_entry,
        entry_scores,
        entry_parent,
        entry_depth,
        entry_tokens,
        start=2,
        depth=2,
        next_lanes=None,
    )
    tokens, parent = draft_tree_finalize(
        entry_scores,
        entry_parent,
        entry_depth,
        entry_tokens,
        torch.tensor([9], device=dev),
        num_nodes=3,
        max_depth=2,
        rank_bits=6,
    )
    for j in range(1, 3):
        assert 0 <= int(parent[0, j]) < j


def test_finalize_keeps_a_parent_tied_with_its_child():
    """A zero log-prob step ties child and parent; the parent (lower id) ranks first."""
    scores = torch.tensor([[-0.5, -0.5, -0.7]], device=DEVICE)
    parent = torch.tensor([[-1, 0, -1]], device=DEVICE)
    depth = torch.tensor([[1, 2, 1]], device=DEVICE)
    tokens = torch.tensor([[10, 11, 12]], device=DEVICE)
    root = torch.tensor([9], device=DEVICE)
    out_tokens, out_parent = draft_tree_finalize(
        scores, parent, depth, tokens, root, num_nodes=2, max_depth=2, rank_bits=6
    )
    assert out_tokens.tolist() == [[9, 10]]
    assert out_parent.tolist() == [[-1, 0]]


@pytest.mark.parametrize("nodes", [2, 17, 64])
def test_tree_ancestry_walks_full_chain(nodes):
    """Depth is not capped: the default chain of N nodes is N - 1 deep."""
    parent = torch.arange(-1, nodes - 1, dtype=torch.int32, device=DEVICE)[None]
    depth = torch.empty_like(parent)
    mask = torch.empty(parent.shape, dtype=torch.int64, device=DEVICE)
    tree_ancestry(parent, depth, mask)
    assert depth[0].tolist() == list(range(nodes))
    got = [m & ((1 << 64) - 1) for m in mask[0].tolist()]
    assert got == [(1 << (i + 1)) - 1 for i in range(nodes)]


@pytest.mark.parametrize(
    "bs,topk,steps,nodes",
    [
        (3, 1, 5, 6),
        (4, 4, 4, 16),
        (2, 8, 5, 64),
        (3, 3, 6, 20),
        # The largest candidate records validation accepts.
        (2, 16, 5, 64),
        (2, 8, 9, 64),
    ],
)
def test_draft_tree_structure(bs, topk, steps, nodes):
    tokens, parent, record = _drive(bs, topk, steps, nodes, vocab=50, seed=nodes)
    depth = torch.empty_like(parent, device=DEVICE)
    mask = torch.empty(parent.shape, dtype=torch.int64, device=DEVICE)
    tree_ancestry(parent.to(DEVICE), depth, mask)
    depth, mask = depth.cpu(), mask.cpu()
    for b in range(bs):
        assert tokens[b, 0] == 7 + b and parent[b, 0] == -1
        paths = {0: ()}
        for j in range(1, nodes):
            p = int(parent[b, j])
            assert 0 <= p < j, "parents precede children"
            paths[j] = paths[p] + (int(tokens[b, j]),)
            assert depth[b, j] == len(paths[j])
            anc, cur = 0, j
            while cur >= 0:
                anc |= 1 << cur
                cur = int(parent[b, cur])
            assert int(mask[b, j]) & ((1 << 64) - 1) == anc  # bit 63 is the int64 sign
        # Kept nodes are the best N - 1 candidates, ties to the earlier candidate.
        ranked = sorted(record[b], key=lambda e: -e[1])
        assert set(paths[j] for j in range(1, nodes)) == {
            e[0] for e in ranked[: nodes - 1]
        }
        # Numbering is depth-first pre-order with children best first.
        score = {e[0]: e[1] for e in record[b]}
        kept = set(paths.values())
        order = []

        def visit(path):
            order.append(path)
            kids = [q for q in kept if len(q) == len(path) + 1 and q[:-1] == path]
            for kid in sorted(kids, key=lambda q: -score[q]):
                visit(kid)

        visit(())
        assert order == [paths[j] for j in range(nodes)]
        # The best path is nodes 1, 2, ...: each node's first kept child is its best.
        node = 0
        while True:
            kids = [j for j in range(1, nodes) if int(parent[b, j]) == node]
            if not kids:
                break
            assert kids[0] == node + 1
            assert max(kids, key=lambda j: score[paths[j]]) == kids[0]
            node = kids[0]
    if topk == 1:
        assert torch.equal(parent, torch.arange(-1, nodes - 1).expand(bs, -1).int())


def _parent_lanes(tree, bs, before_entry):
    """``[bs, K]`` lane each new lane descends from, read off the candidate record."""
    parent_entry = torch.gather(tree.entry_parent[:bs], 1, tree.lane_entry[:bs])
    hits = before_entry[:, None, :] == parent_entry[:, :, None]
    assert torch.all(hits.sum(-1) == 1), "every new lane extends exactly one lane"
    return hits.int().argmax(-1)


def test_nan_scores_keep_lanes_and_tree_valid():
    """A padded request's garbage logits must not leave lanes or nodes unset."""
    bs, topk, steps, nodes, vocab = 2, 4, 3, 8, 50
    tree = DraftTree(bs, topk, steps, nodes, torch.device(DEVICE))
    logits = torch.randn(bs, vocab, device=DEVICE)
    logits[1] = float("nan")
    tree.seed(bs, *logprob_topk(logits, topk))
    lane_logits = torch.randn(bs * topk, vocab, device=DEVICE)
    lane_logits[topk:] = float("nan")
    for step in range(1, steps):
        before = tree.lane_entry[:bs].clone()
        tree.expand(bs, step, *logprob_topk(lane_logits, topk), None)
        _parent_lanes(tree, bs, before)
    tokens, parent = tree.finalize(
        bs, torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    )
    for b in range(bs):
        for j in range(1, nodes):
            assert 0 <= int(parent[b, j]) < j


def test_expand_prepares_next_lanes():
    """Next step's lane masks and hidden rows follow each new lane's parent."""
    torch.manual_seed(3)
    bs, topk, steps, nodes, vocab, width = 3, 4, 4, 12, 64, 40
    tree = DraftTree(bs, topk, steps, nodes, torch.device(DEVICE))
    tree.seed(bs, *logprob_topk(torch.randn(bs, vocab, device=DEVICE), topk))
    lane_mask = (
        torch.ones(topk, dtype=torch.int64, device=DEVICE)
        << torch.arange(topk, device=DEVICE)
    ).repeat(bs, 1)
    for step in range(1, steps):
        hidden_src = torch.randn(bs * topk, width, device=DEVICE)
        hidden_dst = torch.empty_like(hidden_src)
        before = lane_mask.clone()
        before_entry = tree.lane_entry[:bs].clone()
        tree.expand(
            bs,
            step,
            *logprob_topk(torch.randn(bs * topk, vocab, device=DEVICE), topk),
            (lane_mask, hidden_src, hidden_dst),
        )
        parent = _parent_lanes(tree, bs, before_entry)
        own = torch.ones_like(parent) << (
            step * topk + torch.arange(topk, device=DEVICE)
        )
        assert torch.equal(lane_mask, torch.gather(before, 1, parent) | own)
        want = torch.gather(
            hidden_src.view(bs, topk, width),
            1,
            parent[:, :, None].expand(-1, -1, width),
        ).view(bs * topk, width)
        assert torch.equal(hidden_dst, want)
