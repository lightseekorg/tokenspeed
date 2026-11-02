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

"""Block verification probability law, chain layout, sentinels and GPU parity."""

import numpy as np
import pytest
import torch
from tokenspeed_kernel.ops.sampling.block import block_speculative_sampling


def run(p, q, c, u, f, device):
    p = torch.as_tensor(p, dtype=torch.float32, device=device).contiguous()
    q = torch.as_tensor(q, dtype=torch.float32, device=device).contiguous()
    c = torch.as_tensor(c, dtype=torch.int32, device=device)
    b, n, _ = p.shape
    out = torch.full((b * n,), -99, dtype=torch.int32, device=device)
    idx = torch.empty((b, n), dtype=torch.int32, device=device)
    length = torch.empty((b,), dtype=torch.int32, device=device)
    block_speculative_sampling(
        predicts=out,
        accept_index=idx,
        accept_token_num=length,
        candidates=c,
        uniform_samples=torch.as_tensor(u, dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.as_tensor(
            f, dtype=torch.float32, device=device
        ),
        target_probs=p,
        draft_probs=q,
        reject_draft_prob_threshold=2.0,
    )
    return out.cpu(), idx.cpu(), length.cpu()


def test_later_prefix_recovery():
    out, idx, length = run(
        [[[1 / 3, 2 / 3], [2 / 3, 1 / 3], [0.5, 0.5]]],
        [[[2 / 3, 1 / 3], [1 / 3, 2 / 3], [0, 0]]],
        [[0, 0, 0]],
        [[0.9, 0.9, 0.9]],
        [0.8],
        "cpu",
    )
    assert length.tolist() == [2]
    assert out.tolist() == [0, 0, 1]
    assert idx.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("sentinel_at", [0, 1, 2])
def test_missing_proposal_falls_back_to_target(sentinel_at):
    p = torch.tensor([[[0.25, 0.75]] * 3])
    q = p.clone()
    q[:, sentinel_at:] = 3
    out, idx, length = run(p, q, [[0, 1, 1]], [[0.3, 0.3, 0.3]], [0.2], "cpu")
    assert length.item() == sentinel_at
    assert out[sentinel_at].item() == 0
    assert idx[0, : sentinel_at + 1].tolist() == list(range(sentinel_at + 1))
    assert (idx[0, sentinel_at + 1 :] == -1).all()


def test_single_target_draw():
    out, idx, length = run([[[0.2, 0.8]]], [[[3, 3]]], [[0]], [[0.5]], [0.5], "cpu")
    assert out.tolist() == [1] and idx.tolist() == [[0]] and length.tolist() == [0]


def test_zero_coin_does_not_accept_zero_probability():
    # torch.rand can return exactly zero. A target-impossible proposal must
    # still be rejected; a <= test would accept it at this boundary.
    out, idx, length = run(
        [[[0, 1], [0, 1]]], [[[1, 0], [0, 0]]], [[0, 0]], [[0, 0]], [0], "cpu"
    )
    assert out[0].item() == 1
    assert idx.tolist() == [[0, -1]] and length.tolist() == [0]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires GPU"
            ),
        ),
    ],
)
def test_empty_residual_draws_from_target(device):
    # p == q accepts the proposal with probability one, so the residual
    # max(p - q, 0) is empty. A coin of exactly one still rejects it (the
    # rounding case); the replacement must follow p, never token 0.
    out, idx, length = run(
        [[[0, 0.5, 0.5, 0], [0.25, 0.25, 0.25, 0.25]]],
        [[[0, 0.5, 0.5, 0], [0, 0, 0, 0]]],
        [[0, 1]],
        [[1.0, 1.0]],
        [0.75],
        device,
    )
    assert length.tolist() == [0]
    assert idx.tolist() == [[0, -1]]
    assert out[0].item() == 2


def test_one_hot_laws_keep_greedy_prefix_and_target_replacement():
    out, idx, length = run(
        [[[0, 1], [1, 0], [0, 1]]],
        [[[0, 1], [0, 1], [0, 0]]],
        [[0, 1, 1]],
        [[0, 0, 0]],
        [0],
        "cpu",
    )
    assert out[:2].tolist() == [1, 0]
    assert idx.tolist() == [[0, 1, -1]] and length.tolist() == [1]


def test_joint_output_law_monte_carlo():
    # Complete short accepted prefixes with an ordinary target continuation;
    # every two-token output must follow the target's joint distribution.
    rng = np.random.default_rng(123)
    p0 = np.array([0.25, 0.75])
    p1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    q0 = np.array([0.8, 0.2])
    q1 = np.array([[0.2, 0.8], [0.9, 0.1]])
    b = 12000
    c1 = rng.choice(2, size=b, p=q0)
    c2 = (rng.random(b) >= q1[c1, 0]).astype(int)
    p = np.stack([np.broadcast_to(p0, (b, 2)), p1[c1], np.full((b, 2), 0.5)], axis=1)
    q = np.stack([np.broadcast_to(q0, (b, 2)), q1[c1], np.zeros((b, 2))], axis=1)
    out, _, length = run(
        p,
        q,
        np.stack([np.zeros(b), c1, c2], axis=1),
        rng.random((b, 3)),
        rng.random(b),
        "cpu",
    )
    seq = out.reshape(b, 3)[:, :2].numpy().copy()
    zero = length.numpy() == 0
    seq[zero, 1] = (rng.random(zero.sum()) >= p1[seq[zero, 0], 0]).astype(int)
    counts = np.bincount(2 * seq[:, 0] + seq[:, 1], minlength=4) / b
    np.testing.assert_allclose(counts, (p0[:, None] * p1).ravel(), atol=0.012, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("n,v", [(1, 2), (3, 17), (5, 2051), (8, 32001)])
def test_gpu_matches_cpu(n, v):
    rng = np.random.default_rng(45)
    b = 7
    p = rng.dirichlet(np.ones(v), size=(b, n)).astype(np.float32)
    q = rng.dirichlet(np.ones(v), size=(b, n)).astype(np.float32)
    candidates = rng.integers(0, v, size=(b, n), dtype=np.int32)
    q[0] = 3
    u = rng.random((b, n), dtype=np.float32)
    f = rng.random(b, dtype=np.float32)
    cpu = run(p, q, candidates, u, f, "cpu")
    gpu = run(p, q, candidates, u, f, "cuda")
    for a, b_ in zip(cpu, gpu):
        torch.testing.assert_close(a, b_, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_gpu_graph_replay_reads_refreshed_probabilities_and_coins():
    # Capture and replay share the eager kernel, with in-place refreshed inputs.
    # This test must execute on a GPU before graph compatibility is claimed.
    b, n, v = 2, 3, 17
    gen = torch.Generator().manual_seed(239)
    p = torch.rand((b, n, v), generator=gen)
    p /= p.sum(-1, keepdim=True)
    q = torch.rand((b, n, v), generator=gen)
    q /= q.sum(-1, keepdim=True)
    c = torch.randint(v, (b, n), generator=gen, dtype=torch.int32)
    u = torch.rand((b, n), generator=gen)
    f = torch.rand((b,), generator=gen)
    args = {
        "predicts": torch.full((b * n,), -99, device="cuda", dtype=torch.int32),
        "accept_index": torch.empty((b, n), device="cuda", dtype=torch.int32),
        "accept_token_num": torch.empty((b,), device="cuda", dtype=torch.int32),
        "candidates": c.cuda(),
        "uniform_samples": u.cuda(),
        "uniform_samples_for_final_sampling": f.cuda(),
        "target_probs": p.cuda(),
        "draft_probs": q.cuda(),
        "reject_draft_prob_threshold": 2.0,
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            block_speculative_sampling(**args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        block_speculative_sampling(**args)
    for sentinel in (False, True):
        if sentinel:
            q[0] = 3
        u = torch.rand((b, n), generator=gen)
        f = torch.rand((b,), generator=gen)
        args["draft_probs"].copy_(q)
        args["uniform_samples"].copy_(u)
        args["uniform_samples_for_final_sampling"].copy_(f)
        args["predicts"].fill_(-99)
        graph.replay()
        expected = run(p, q, c, u, f, "cpu")
        for name, value in zip(
            ("predicts", "accept_index", "accept_token_num"), expected
        ):
            torch.testing.assert_close(args[name].cpu(), value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_gpu_batch_size_and_strides_do_not_recompile():
    # N and V are server geometry; the batch size and coin/candidate strides
    # change every step and must stay runtime values.
    # The shared test utilities need a GPU platform at import time.
    from tokenspeed_kernel.ops.sampling.triton import block as triton_block
    from utils import assert_no_triton_compile

    rng = np.random.default_rng(47)
    n, v = 5, 2051

    def step(b):
        p = rng.dirichlet(np.ones(v), size=(b, n)).astype(np.float32)
        q = rng.dirichlet(np.ones(v), size=(b, n)).astype(np.float32)
        c = rng.integers(0, v, size=(b, n), dtype=np.int32)
        run(p, q, c, rng.random((b, n)), rng.random(b), "cuda")

    step(3)
    with assert_no_triton_compile(
        triton_block._prefix_ratios,
        triton_block._residual_mass,
        triton_block._select_prefix,
        triton_block._sample_residual,
    ):
        for b in (1, 16, 17, 64):
            step(b)
