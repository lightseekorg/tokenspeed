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

"""sonic verifies draft trees on the target's own draws.

Every node row draws once, as ``sample`` would draw there: keyed by the node's
depth, with the tokens on its path counted for penalties. Walking from the
root, the child holding its parent's draw is accepted and the first draw no
child holds is the bonus token, so the emitted tokens follow the target for
any tree and a child is accepted at its target probability. The tokens
emitted join the decode counts that penalties read.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="sonic's kernels need NVIDIA's libdevice (fast_logf).",
)

from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.sonic import (  # noqa: E402
    SonicSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.sampling.tree_verify import TreeVerifyBatch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

VOCAB = 1024
REAL_VOCAB = 151936
POOL = 8
BS = 4
# The root, two children, two grandchildren under the first child.
PARENTS = [-1, 0, 0, 1, 1]
DEPTHS = [0, 1, 1, 2, 2]
N = len(PARENTS)


def _backend(vocab: int = VOCAB, logprobs: bool = False) -> SonicSamplingBackend:
    return SonicSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            enable_output_logprobs=logprobs,
            max_bs=BS,
            max_draft_tokens_per_req=N,
            draft_tree=True,
            max_req_pool_size=POOL,
            vocab_size=vocab,
            device="cuda",
        )
    )


def _sp(rid: str, vocab: int = VOCAB, **overrides) -> SamplingParams:
    params = dict(temperature=1.0, top_k=-1, top_p=1.0)
    params.update(overrides)
    sp = SamplingParams(**params)
    sp.verify(vocab)
    sp.resolve_seed(rid)
    sp.normalize(None)
    return sp


def _prepare(backend: SonicSamplingBackend, sps: list[SamplingParams]) -> torch.Tensor:
    slots = list(range(1, len(sps) + 1))
    backend.prepare_step(
        request_ids=[f"rid_{sp.seed}_{i}" for i, sp in enumerate(sps)],
        request_pool_indices=slots,
        sampling_params_list=sps,
        num_tokens_per_req=N,
    )
    return torch.tensor(slots, dtype=torch.int64, device="cuda")


def _info(req: torch.Tensor, offsets: torch.Tensor | None = None) -> SamplingBatchInfo:
    if offsets is None:
        offsets = torch.arange(100, 100 + POOL + 1, dtype=torch.int32, device="cuda")
    return SamplingBatchInfo(
        req_pool_indices=req,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
    )


def _tree(bs: int = BS) -> TreeVerifyBatch:
    parents = torch.tensor(PARENTS, dtype=torch.int32, device="cuda")
    depths = torch.tensor(DEPTHS, dtype=torch.int32, device="cuda")
    return TreeVerifyBatch(
        parents=parents.repeat(bs, 1), depths=depths.repeat(bs, 1), draft_q=None
    )


def _peaks(rows: int, vocab: int = VOCAB) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 logits whose rows each peak at a distinct token, and those tokens."""
    logits = torch.randn(rows, vocab, device="cuda")
    peaks = torch.randperm(vocab, device="cuda")[:rows]
    logits[torch.arange(rows, device="cuda"), peaks] = 12.0
    return logits.to(torch.bfloat16), peaks.int()


def _other(token: int) -> int:
    return (token + 1) % VOCAB


def test_greedy_trees_walk_the_argmax_path_and_count_it():
    backend = _backend()
    req = _prepare(backend, [_sp(f"g{i}", temperature=0.0) for i in range(BS)])
    logits, peaks = _peaks(BS * N)
    peak = peaks.view(BS, N).tolist()
    candidates = torch.zeros(BS, N, dtype=torch.int32)
    for b in range(BS):
        candidates[b] = torch.tensor([_other(peak[b][0])] * N)
    # Request 0 accepts nodes 1 then 4, request 1 node 2, request 2 none, request 3 nodes 1 then 3.
    candidates[0, 1], candidates[0, 2] = peak[0][0], _other(peak[0][0])
    candidates[0, 3], candidates[0, 4] = _other(peak[0][1]), peak[0][1]
    candidates[1, 1], candidates[1, 2] = _other(peak[1][0]), peak[1][0]
    candidates[1, 3], candidates[1, 4] = peak[1][1], _other(peak[1][1])
    candidates[2, 1], candidates[2, 2] = _other(peak[2][0]), _other(_other(peak[2][0]))
    candidates[3, 1], candidates[3, 2] = peak[3][0], _other(peak[3][0])
    candidates[3, 3], candidates[3, 4] = peak[3][1], _other(peak[3][1])

    predict, accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits),
        _info(req),
        candidates.cuda(),
        tree=_tree(),
    )
    paths = [[0, 1, 4], [0, 2], [0], [0, 1, 3]]
    assert accept.tolist() == [len(p) for p in paths]
    predict = predict.view(BS, N).tolist()
    accepted = backend.accepted_path(BS, N).tolist()
    for b, path in enumerate(paths):
        assert predict[b][: len(path)] == [peak[b][node] for node in path]
        assert accepted[b] == path + [-1] * (N - len(path))
    # Each emitted token counts once for its slot's penalties.
    counts = backend.state.decode_counts
    for b, path in enumerate(paths):
        assert int(counts[req[b]].sum()) == len(path)
        for node in path:
            assert int(counts[req[b], peak[b][node]]) == 1


def test_a_padded_verify_reads_back_at_the_live_batch_size():
    """A captured graph verifies its padded batch; the executor reads the path back at the live batch size."""
    backend = _backend()
    req = _prepare(backend, [_sp(f"pad{i}", temperature=0.0) for i in range(BS)])
    logits, peaks = _peaks(BS * N)
    candidates = torch.randint(0, VOCAB, (BS, N), dtype=torch.int32, device="cuda")
    candidates[:, 1] = peaks.view(BS, N)[:, 0]
    predict, accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits),
        _info(req),
        candidates,
        tree=_tree(),
    )
    path = backend.accepted_path(BS, N).clone()
    assert path[:, 1].tolist() == [1] * BS
    for live in range(1, BS):
        assert torch.equal(backend.accepted_path(live, N), path[:live])


def test_penalized_greedy_trees_count_each_node_path():
    """Frequency penalty 2.0 over 10.0 > 9.5 > 9.0: every node row peaks at t0, so a node below t0 draws t1, and one below t0 and t1 draws t2."""
    t0, t1, t2 = 10, 20, 30
    backend = _backend()
    req = _prepare(
        backend,
        [_sp(f"p{i}", temperature=0.0, frequency_penalty=2.0) for i in range(BS)],
    )
    logits = torch.full((BS * N, VOCAB), -5.0, device="cuda")
    logits[:, [t0, t1, t2]] = torch.tensor([10.0, 9.5, 9.0], device="cuda")
    candidates = torch.full((BS, N), 500, dtype=torch.int32, device="cuda")
    candidates[:, 1], candidates[:, 2] = t0, t1
    candidates[:, 3], candidates[:, 4] = 600, t1

    predict, accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.to(torch.bfloat16)),
        _info(req),
        candidates,
        tree=_tree(),
    )
    # The root draws t0 (node 1), node 1 draws t1 (node 4), node 4 draws t2.
    assert accept.tolist() == [3] * BS
    assert predict.view(BS, N)[:, :3].tolist() == [[t0, t1, t2]] * BS
    assert backend.accepted_path(BS, N).tolist() == [[0, 1, 4, -1, -1]] * BS


def _chi2_deviate(counts: torch.Tensor, probs: torch.Tensor, n: int) -> float:
    """Normal deviate of the chi-square statistic over cells expecting >= 20 draws."""
    keep = probs * n >= 20
    obs, exp = counts[keep].double(), probs[keep].double() * n
    dof = int(keep.sum().item()) - 1
    return ((((obs - exp) ** 2) / exp).sum().item() - dof) / math.sqrt(2 * dof)


@pytest.mark.parametrize("top_k", [-1, 16], ids=["whole", "top_k"])
def test_sampled_trees_emit_the_target_law(top_k):
    """Each node row holds one law p: the first token follows p, child i is accepted at p(token i), and the token after an accepted child follows p again."""
    torch.manual_seed(5)
    iters = 3000
    backend = _backend()
    req = _prepare(backend, [_sp(f"s{i}", top_k=top_k) for i in range(BS)])
    row = (torch.randn(VOCAB) * 1.5).to(torch.bfloat16)
    row[64:] = -40.0
    law = torch.softmax(row.float(), -1)
    if top_k > 0:
        cut = law.topk(top_k).indices
        law = torch.zeros_like(law).index_fill_(0, cut, 1.0) * law
        law /= law.sum()
    top = law.topk(4).indices.tolist()
    logits = row.cuda().expand(BS * N, VOCAB).contiguous()
    candidates = torch.zeros(BS, N, dtype=torch.int32, device="cuda")
    candidates[:, 1], candidates[:, 2] = top[0], top[1]
    candidates[:, 3], candidates[:, 4] = top[2], top[3]
    offsets = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    info = _info(req, offsets)
    tree = _tree()

    first = torch.zeros(VOCAB, dtype=torch.long, device="cuda")
    second = torch.zeros(VOCAB, dtype=torch.long, device="cuda")
    took = torch.zeros(N, dtype=torch.long, device="cuda")
    for step in range(iters):
        offsets.fill_(step * N)
        predict, accept = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits), info, candidates, tree=tree
        )
        predict = predict.view(BS, N).long()
        first.scatter_add_(
            0, predict[:, 0], torch.ones(BS, dtype=torch.long, device="cuda")
        )
        path = backend.accepted_path(BS, N).long()
        took.scatter_add_(0, path[:, 1].clamp(min=0), (path[:, 1] > 0).long())
        deep = (path[:, 1] == 1).long()
        second.scatter_add_(0, predict[:, 1] * deep, deep)
    total = iters * BS
    assert abs(_chi2_deviate(first.cpu(), law, total)) < 5.0
    for node, token in ((1, top[0]), (2, top[1])):
        p = law[token].item()
        rate = took[node].item() / total
        assert abs(rate - p) < 4 * math.sqrt(p * (1 - p) / total) + 0.005, (
            node,
            rate,
            p,
        )
    below = int(took[1].item())
    assert abs(_chi2_deviate(second.cpu(), law, below)) < 5.0


def test_logprobs_score_the_accepted_path_rows():
    backend = _backend(logprobs=True)
    req = _prepare(backend, [_sp(f"l{i}", temperature=0.0) for i in range(BS)])
    logits, peaks = _peaks(BS * N)
    peak = peaks.view(BS, N).tolist()
    candidates = torch.zeros(BS, N, dtype=torch.int32)
    for b in range(BS):
        candidates[b, 1], candidates[b, 2] = _other(peak[b][0]), peak[b][0]
    out = LogitsProcessorOutput(next_token_logits=logits)
    predict, accept = backend.verify(out, _info(req), candidates.cuda(), tree=_tree())
    # Every request accepts node 2: the logprobs score rows 0 and 2.
    assert accept.tolist() == [2] * BS
    rows = torch.tensor([[b * N, b * N + 2] for b in range(BS)], device="cuda")
    reference = torch.log_softmax(logits.float(), -1)
    expected = reference[rows.view(-1)].gather(
        1, predict.view(BS, N)[:, :2].reshape(-1, 1).long()
    )
    logprobs = out.next_token_logprobs.float().view(BS, N)[:, :2].reshape(-1, 1)
    torch.testing.assert_close(logprobs, expected, atol=1e-2, rtol=1e-2)


def test_tree_cuda_graph_replay_matches_eager():
    torch.manual_seed(9)
    backend = _backend()
    sps = [
        _sp("cg0", temperature=0.0),
        _sp("cs1", top_k=16),
        _sp("cs2", top_p=0.9),
        _sp("cs3"),
    ]
    backend.prepare_capture(bs=BS, num_tokens_per_req=N)
    logits_buf = torch.zeros(BS * N, VOCAB, device="cuda", dtype=torch.bfloat16)
    cand_buf = torch.zeros(BS, N, dtype=torch.int32, device="cuda")
    req_buf = torch.zeros(BS, dtype=torch.int64, device="cuda")
    tree = _tree()
    info = _info(req_buf)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.verify(
                LogitsProcessorOutput(next_token_logits=logits_buf),
                info,
                cand_buf,
                tree=tree,
            )
    torch.cuda.current_stream().wait_stream(stream)
    backend.reset_capture_state()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g_predict, g_accept = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits_buf),
            info,
            cand_buf,
            tree=tree,
        )

    req = _prepare(backend, sps)
    logits, peaks = _peaks(BS * N)
    candidates = torch.randint(0, VOCAB, (BS, N), dtype=torch.int32, device="cuda")
    candidates[:, 1] = peaks.view(BS, N)[:, 0]
    before = backend.state.decode_counts.clone()
    e_predict, e_accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info(req),
        candidates,
        tree=tree,
    )
    e_predict, e_accept = e_predict.clone(), e_accept.clone()
    e_path = backend.accepted_path(BS, N).clone()
    eager = backend.state.decode_counts.clone()

    # Same inputs through the graph: the same outputs, counted once more.
    logits_buf.copy_(logits)
    cand_buf.copy_(candidates)
    req_buf.copy_(req)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(g_predict, e_predict)
    assert torch.equal(g_accept, e_accept)
    assert torch.equal(backend.accepted_path(BS, N), e_path)
    assert torch.equal(backend.state.decode_counts - eager, eager - before)
    assert int((eager - before).sum()) == int(e_accept.sum())


def _kernel_test_utils():
    """tokenspeed-kernel's test helpers, for ``assert_no_triton_compile``."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(os.path.dirname(root), "tokenspeed-kernel", "test", "utils.py")
    spec = importlib.util.spec_from_file_location("sonic_kernel_test_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_tree_launches_compile_nothing_after_warm_up(dtype):
    """Warm-up compiles what the tree verify of every batch size launches."""
    from tokenspeed_kernel.ops.sampling import sonic
    from tokenspeed_kernel.ops.sampling.triton import tree_verify
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import prologue, singular

    utils = _kernel_test_utils()
    backend = _backend(REAL_VOCAB)
    sps = [_sp(f"w{i}", REAL_VOCAB, temperature=0.7, top_p=0.9) for i in range(BS)]
    req = _prepare(backend, sps)
    logits = torch.randn(BS * N, REAL_VOCAB, device="cuda").to(dtype)
    candidates = torch.randint(0, REAL_VOCAB, (BS, N), dtype=torch.int32, device="cuda")
    # fp32 kernels warm on the first fp32 launch.
    backend._check_logits(logits, None)
    kernels = (
        prologue.bitpacked_reduction_kernel,
        prologue.whole_reduction_kernel,
        singular.cumulative_selection_kernel,
        sonic._tree_paths_kernel,
        sonic._count_emitted_kernel,
        tree_verify._verify_tree_kernel,
    )
    info = _info(req)
    with utils.assert_no_triton_compile(*kernels):
        for bs in range(1, BS + 1):
            backend.verify(
                LogitsProcessorOutput(next_token_logits=logits[: bs * N]),
                SamplingBatchInfo(
                    req_pool_indices=req[:bs],
                    valid_cache_lengths=info.valid_cache_lengths,
                    vocab_size=REAL_VOCAB,
                    device="cuda",
                ),
                candidates[:bs],
                tree=_tree(bs),
            )
    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
