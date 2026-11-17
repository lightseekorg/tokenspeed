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

"""DeepInfer packed-chain rejection sampling and stateless RNG tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from tokenspeed_kernel.platform import current_platform

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_POSITION_STRIDE = 10000019
_TARGET, _DRAFT, _ACCEPT, _REJECT, _BONUS = range(5)
_RESIDUAL_PENALTY = math.exp(-5e-5)


def _mulhilo(a: int, b: int) -> tuple[int, int]:
    product = a * b
    return (product >> 32) & _MASK32, product & _MASK32


def _philox_uniform(seed: int, position: int, rng_tag: int) -> float:
    """Reference Philox4x32-10 stream used by the packed sampling API."""
    position &= _MASK64
    c0, c1 = position & _MASK32, position >> 32
    c2, c3 = rng_tag & _MASK32, 0
    k0, k1 = seed & _MASK32, (seed >> 32) & _MASK32
    for _ in range(10):
        hi0, lo0 = _mulhilo(_M0, c0)
        hi1, lo1 = _mulhilo(_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & _MASK32,
            lo1,
            (hi0 ^ c3 ^ k1) & _MASK32,
            lo0,
        )
        k0 = (k0 + _W0) & _MASK32
        k1 = (k1 + _W1) & _MASK32
    return (c0 >> 8) * (2.0**-24)


def _draw(
    probs: torch.Tensor, uniform: float, fallback_probs: torch.Tensor | None = None
) -> int:
    """Sample with strict CDF > u*mass and lowest-argmax zero-mass fallback."""
    mass = sum(probs.tolist())
    if mass <= 0:
        return int((probs if fallback_probs is None else fallback_probs).argmax())
    threshold = uniform * mass
    cdf = 0.0
    for token, prob in enumerate(probs.tolist()):
        cdf += prob
        if cdf > threshold:
            return token
    return int(probs.argmax())


def _oracle_sample_probs(
    probs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor, rng_tag: int
) -> torch.Tensor:
    return torch.tensor(
        [
            _draw(
                probs[row],
                _philox_uniform(int(seeds[row]), int(positions[row]), rng_tag),
            )
            for row in range(probs.shape[0])
        ],
        dtype=torch.int32,
    )


def _oracle_verify(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    query_start_loc: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU reference for packed DeepInfer verification."""
    batch, width = candidates.shape
    predicts = torch.zeros(batch * width, dtype=torch.int32)
    accept_index = torch.full((batch, width), -1, dtype=torch.int32)
    accept_length = torch.zeros(batch, dtype=torch.int32)
    for req in range(batch):
        start = int(query_start_loc[req])
        query_count = int(query_start_loc[req + 1]) - start
        for step in range(max(query_count - 1, 0)):
            # Candidate column zero is the anchor; step zero verifies column one.
            col = step + 1
            token = int(candidates[req, col])
            p = float(target_probs[start + step, token])
            q = float(draft_probs[req, step, token])
            position = (int(positions[req]) + step * _POSITION_STRIDE) & _MASK64
            u_accept = _philox_uniform(int(seeds[req]), position, _ACCEPT)
            ratio = (
                1.0 if q > 2.0 else (1.0 if q == 0 and p > 0 else p / q if q else 0.0)
            )
            if q <= 2.0 and u_accept < min(1.0, ratio):
                predicts[req * width + step] = token
                accept_index[req, step] = req * width + step
                continue
            if q > 2.0:
                residual = target_probs[start + step]
            else:
                residual = torch.clamp(
                    target_probs[start + step]
                    - _RESIDUAL_PENALTY * draft_probs[req, step],
                    min=0,
                )
            u_reject = _philox_uniform(int(seeds[req]), position, _REJECT)
            predicts[req * width + step] = _draw(
                residual, u_reject, target_probs[start + step]
            )
            accept_index[req, step] = req * width + step
            accept_length[req] = step + 1
            break
        else:
            if query_count == 0:
                continue
            step = query_count - 1
            position = (int(positions[req]) + step * _POSITION_STRIDE) & _MASK64
            predicts[req * width + step] = _draw(
                target_probs[start + step],
                _philox_uniform(int(seeds[req]), position, _BONUS),
            )
            accept_index[req, step] = req * width + step
            accept_length[req] = step + 1
    return predicts, accept_index, accept_length


def _api():
    return pytest.importorskip("tokenspeed_kernel.ops.sampling.packed_verify")


def _need_cuda() -> None:
    if not torch.cuda.is_available() or not current_platform().is_hopper_plus:
        pytest.skip("NVIDIA SM90+ GPU is required for packed sampling kernels")


def _run_verify(api, inputs, *, draft_pool_indices=None):
    candidates, target, draft, query_start_loc, seeds, positions = inputs
    batch, width = candidates.shape
    predicts = torch.full((batch * width,), -17, device="cuda", dtype=torch.int32)
    accept_index = torch.full((batch, width), -17, device="cuda", dtype=torch.int32)
    accept_length = torch.full((batch,), -17, device="cuda", dtype=torch.int32)
    kwargs = {}
    if draft_pool_indices is not None:
        kwargs["draft_pool_indices"] = draft_pool_indices
    api.verify_packed_chain(
        reject_draft_prob_threshold=2.0,
        predicts=predicts,
        accept_index=accept_index,
        accept_length=accept_length,
        candidates=candidates,
        target_probs=target,
        draft_probs=draft,
        query_start_loc=query_start_loc,
        seeds=seeds,
        positions=positions,
        **kwargs,
    )
    return predicts, accept_index, accept_length


def _run_verify_into(api, inputs, outputs):
    candidates, target, draft, query_start_loc, seeds, positions = inputs
    predicts, accept_index, accept_length = outputs
    api.verify_packed_chain(
        reject_draft_prob_threshold=2.0,
        predicts=predicts,
        accept_index=accept_index,
        accept_length=accept_length,
        candidates=candidates,
        target_probs=target,
        draft_probs=draft,
        query_start_loc=query_start_loc,
        seeds=seeds,
        positions=positions,
    )
    return outputs


def _make_chain_inputs(batch: int = 8, width: int = 4):
    device = "cuda"
    query_counts = [4, 2, 1, 3, 4, 1, 2, 0][:batch]
    starts = [0]
    for count in query_counts:
        starts.append(starts[-1] + count)
    rows, vocab = starts[-1] + 2, 8
    target = torch.zeros((rows, vocab), device=device, dtype=torch.float32)
    target[:, :4] = torch.tensor([0.125, 0.5, 0.25, 0.125], device=device)
    target[-2:] = float("nan")
    draft = torch.zeros((batch, width, vocab), device=device, dtype=torch.float32)
    draft[:, :, :4] = torch.tensor([0.5, 0.25, 0.125, 0.125], device=device)
    if width > 2:
        draft[:, 1, :4] = torch.tensor([0.5, 0.0, 0.5, 0.0], device=device)
    if width > 3:
        draft[:, 2, :4] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    candidates = torch.tensor(
        [[4, 0, 2, 1] for _ in range(batch)], device=device, dtype=torch.int32
    )
    for req, count in enumerate(query_counts):
        if count > 1 and req == 1:
            draft[req, 0].fill_(0)
            draft[req, 0, 0] = 2.5
    seeds = torch.arange(batch, device=device, dtype=torch.int64) * 107 + 91
    positions = torch.arange(batch, device=device, dtype=torch.int64) * 1000003 + 13
    patterns = {
        0: (0, [True, True, True]),
        3: (1, [True, False]),
        4: (2, [True, True, False]),
    }
    for req, (last_step, decisions) in patterns.items():
        if req >= batch or query_counts[req] <= last_step + 1:
            continue
        pos = int(positions[req])
        ratios = [0.25, 0.5, 0.5][: len(decisions)]
        for seed in range(10000):
            actual = [
                _philox_uniform(seed, pos + step * _POSITION_STRIDE, _ACCEPT) < ratio
                for step, ratio in enumerate(ratios)
            ]
            if actual == decisions:
                seeds[req] = seed
                break
        else:
            raise AssertionError(
                f"failed to find deterministic decision pattern {decisions}"
            )
    return (
        candidates,
        target,
        draft,
        torch.tensor(starts, device=device, dtype=torch.int64),
        seeds,
        positions,
    )


def test_philox_reference_matches_known_vector() -> None:
    assert _philox_uniform(0, 0, 0) == (0x6627E8D5 >> 8) * (2.0**-24)


def test_oracle_zero_mass_fallback_uses_lowest_target_argmax() -> None:
    assert _draw(torch.zeros(4), 0.4) == 0
    assert _draw(torch.zeros(4), 0.4, torch.tensor([0.1, 0.7, 0.7, 0.0])) == 1


def test_oracle_cdf_roundoff_fallback_uses_lowest_weight_argmax() -> None:
    # Force a missed CDF crossing to exercise the roundoff fallback.
    assert _draw(torch.tensor([0.25, 0.5, 0.5, 0.0]), 1.0) == 1


def test_cpu_oracle_handles_empty_request_and_residual_walk() -> None:
    target = torch.tensor(
        [[0.125, 0.5, 0.25, 0.125], [0.25, 0.25, 0.25, 0.25]],
        dtype=torch.float32,
    )
    draft = torch.tensor(
        [
            [[0.5, 0.25, 0.125, 0.125], [0.5, 0.25, 0.125, 0.125]],
            [[0.5, 0.25, 0.125, 0.125], [0.5, 0.25, 0.125, 0.125]],
        ],
        dtype=torch.float32,
    )
    result = _oracle_verify(
        candidates=torch.tensor([[4, 0], [4, 0]], dtype=torch.int32),
        target_probs=target,
        draft_probs=draft,
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int64),
        seeds=torch.tensor([17, 18], dtype=torch.int64),
        positions=torch.tensor([9, 10], dtype=torch.int64),
    )
    assert result[2].tolist() == [1, 0]
    assert result[1].tolist() == [[0, -1], [-1, -1]]
    assert result[0].tolist() == [2, 0, 0, 0]


@pytest.mark.parametrize(("rows", "vocab"), [(257, 8), (32, 4099)])
def test_sample_probs_matches_stateless_cpu_philox(rows: int, vocab: int) -> None:
    _need_cuda()
    api = _api()
    probs = torch.zeros((rows, vocab), device="cuda", dtype=torch.float32)
    tokens = [0, 1, 2, 3] if vocab == 8 else [0, 1023, 1024, 4098]
    probs[:, tokens] = torch.tensor([0.125, 0.5, 0.25, 0.125], device="cuda")
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) * 31 + 7
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 1009 - 5
    output = torch.empty(rows, device="cuda", dtype=torch.int32)
    before = (probs.clone(), seeds.clone(), positions.clone())
    api.sample_probs(
        probs=probs, seeds=seeds, positions=positions, output=output, rng_tag=_TARGET
    )
    expected = _oracle_sample_probs(
        probs.cpu(), seeds.cpu(), positions.cpu(), _TARGET
    ).to("cuda")
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    for old, new in zip(before, (probs, seeds, positions), strict=True):
        assert torch.equal(old, new)


def test_sample_probs_separates_rng_namespaces() -> None:
    _need_cuda()
    api = _api()
    rows = 32
    probs = (
        torch.tensor([0.25, 0.25, 0.25, 0.25], device="cuda")
        .expand(rows, -1)
        .contiguous()
    )
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + 101
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 13
    for tag in (_TARGET, _DRAFT, _ACCEPT, _REJECT, _BONUS):
        output = torch.empty(rows, device="cuda", dtype=torch.int32)
        api.sample_probs(
            probs=probs, seeds=seeds, positions=positions, output=output, rng_tag=tag
        )
        expected = _oracle_sample_probs(
            probs.cpu(), seeds.cpu(), positions.cpu(), tag
        ).to("cuda")
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_sample_probs_tracks_probability_distribution() -> None:
    _need_cuda()
    api = _api()
    rows = 10000
    row_probs = torch.tensor([0.125, 0.5, 0.25, 0.125], device="cuda")
    probs = row_probs.expand(rows, -1).contiguous()
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + 700
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 37
    output = torch.empty(rows, device="cuda", dtype=torch.int32)
    api.sample_probs(
        probs=probs, seeds=seeds, positions=positions, output=output, rng_tag=_TARGET
    )
    counts = torch.bincount(output.long(), minlength=4).float()
    expected = rows * row_probs
    tolerance = 7 * torch.sqrt(rows * row_probs * (1 - row_probs)) + 2
    assert bool(((counts - expected).abs() <= tolerance).all())


def test_sample_probs_zero_mass_rows_fall_back_to_token_zero() -> None:
    _need_cuda()
    api = _api()
    rows, vocab = 7, 11
    probs = torch.zeros((rows, vocab), device="cuda", dtype=torch.float32)
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 17
    output = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    api.sample_probs(
        probs=probs,
        seeds=seeds,
        positions=positions,
        output=output,
        rng_tag=_TARGET,
    )
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.parametrize(
    ("vocab", "seed", "position"),
    [(1024, 3822200, 123), (129280, 3822200, 123)],
)
def test_sample_probs_never_returns_zero_mass_token_at_cdf_boundary(
    vocab: int, seed: int, position: int
) -> None:
    _need_cuda()
    api = _api()
    generator = np.random.default_rng(7 if vocab == 1024 else 2)
    row = generator.random(vocab, dtype=np.float32)
    row[-1] = 0
    row /= row.sum(dtype=np.float32)
    probs = torch.from_numpy(row).to(device="cuda").view(1, -1).contiguous()
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    api.sample_probs(
        probs=probs,
        seeds=torch.tensor([seed], device="cuda", dtype=torch.int64),
        positions=torch.tensor([position], device="cuda", dtype=torch.int64),
        output=output,
        rng_tag=_BONUS,
    )
    assert probs[0, -1].item() == 0
    assert output.item() != vocab - 1


def test_verify_packed_chain_matches_cpu_oracle_preserves_inputs() -> None:
    _need_cuda()
    api = _api()
    inputs = _make_chain_inputs()
    before = tuple(t.clone() for t in inputs)
    actual = _run_verify(api, inputs)
    expected = _oracle_verify(
        candidates=inputs[0].cpu(),
        target_probs=inputs[1].cpu(),
        draft_probs=inputs[2].cpu(),
        query_start_loc=inputs[3].cpu(),
        seeds=inputs[4].cpu(),
        positions=inputs[5].cpu(),
    )
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)
    assert expected[2].tolist()[0] == 4  # all draft tokens accepted, then bonus
    assert expected[2].tolist()[1] == 1  # Unavailable proposal rejects immediately.
    assert expected[2].tolist()[3] == 2  # rejection after one accepted draft
    assert expected[2].tolist()[4] == 3  # rejection after two accepted drafts
    assert expected[2].tolist()[7] == 0  # empty CSR request
    for old, new in zip(before, inputs, strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0, equal_nan=True)


def test_verify_packed_chain_uses_reordered_pool_indices() -> None:
    _need_cuda()
    api = _api()
    width, vocab, pool_rows = 4, 8, 9
    p = torch.tensor([0.125, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0], device="cuda")
    target = p.expand(7, -1).contiguous()
    draft_pool = torch.full(
        (pool_rows, width, vocab), 2.5, device="cuda", dtype=torch.float32
    )
    draft_pool[7] = p
    pool_indices = torch.tensor([7, 2, 5], device="cuda", dtype=torch.int64)
    starts = torch.tensor([0, 4, 6, 7], device="cuda", dtype=torch.int64)
    candidates = torch.tensor(
        [[7, 0, 2, 1], [7, 0, 2, 1], [7, 0, 2, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    seeds = torch.tensor([71, 27, 55], device="cuda", dtype=torch.int64)
    positions = torch.tensor([101, 203, 307], device="cuda", dtype=torch.int64)
    actual = _run_verify(
        api,
        (candidates, target, draft_pool, starts, seeds, positions),
        draft_pool_indices=pool_indices,
    )
    expected = _oracle_verify(
        candidates=candidates.cpu(),
        target_probs=target.cpu(),
        draft_probs=draft_pool.index_select(0, pool_indices).cpu(),
        query_start_loc=starts.cpu(),
        seeds=seeds.cpu(),
        positions=positions.cpu(),
    )
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)
    assert actual[2].tolist() == [4, 1, 1]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.verify_packed_chain(
            reject_draft_prob_threshold=2.0,
            predicts=actual[0],
            accept_index=actual[1],
            accept_length=actual[2],
            candidates=candidates,
            target_probs=target,
            draft_probs=draft_pool,
            draft_pool_indices=pool_indices,
            query_start_loc=starts,
            seeds=seeds,
            positions=positions,
        )
    pool_indices.copy_(torch.tensor([5, 7, 2], device="cuda", dtype=torch.int64))
    graph.replay()
    expected = _oracle_verify(
        candidates=candidates.cpu(),
        target_probs=target.cpu(),
        draft_probs=draft_pool.index_select(0, pool_indices).cpu(),
        query_start_loc=starts.cpu(),
        seeds=seeds.cpu(),
        positions=positions.cpu(),
    )
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)

    from utils import assert_no_cute_compile

    with assert_no_cute_compile():
        for pool_rows, mapping in ((11, [7, 2, 5]), (13, [5, 7, 2])):
            larger_pool = torch.full(
                (pool_rows, width, vocab), 2.5, device="cuda", dtype=torch.float32
            )
            larger_pool[: draft_pool.shape[0]].copy_(draft_pool)
            indices = torch.tensor(mapping, device="cuda", dtype=torch.int64)
            outputs = _run_verify(
                api,
                (candidates, target, larger_pool, starts, seeds, positions),
                draft_pool_indices=indices,
            )
            expected = _oracle_verify(
                candidates=candidates.cpu(),
                target_probs=target.cpu(),
                draft_probs=larger_pool.index_select(0, indices).cpu(),
                query_start_loc=starts.cpu(),
                seeds=seeds.cpu(),
                positions=positions.cpu(),
            )
            for got, ref in zip(outputs, expected, strict=True):
                torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)


def test_verify_rejection_never_emits_zero_residual_mass_token() -> None:
    _need_cuda()
    api = _api()
    vocab = 1024
    generator = np.random.default_rng(1)
    target_row = generator.random(vocab, dtype=np.float32)
    target_row[0] = 0
    target_row[-1] = 0
    target_row /= target_row.sum(dtype=np.float32)
    target_row *= 0.5
    target_row[0] = 0.5
    p = torch.from_numpy(target_row).to(device="cuda")
    q = torch.zeros((1, 2, vocab), device="cuda", dtype=torch.float32)
    q[0, 0, 0] = 1.0
    target = p.expand(2, -1).contiguous()
    candidates = torch.tensor([[vocab - 1, 0]], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    seeds = torch.tensor([20115333], device="cuda", dtype=torch.int64)
    positions = torch.tensor([123], device="cuda", dtype=torch.int64)
    actual = _run_verify(api, (candidates, target, q, starts, seeds, positions))
    assert actual[2].tolist() == [1]
    residual = torch.clamp(p - _RESIDUAL_PENALTY * q[0, 0], min=0)
    token = actual[0][0].item()
    assert residual[token].item() > 0


def test_verify_packed_chain_samples_conditional_residual() -> None:
    _need_cuda()
    api = _api()
    rows, width = 10000, 2
    vocab = 4
    p = torch.tensor([0.125, 0.5, 0.25, 0.125], device="cuda")
    q = torch.tensor([0.5, 0.25, 0.125, 0.125], device="cuda")
    target = p.expand(rows * width, -1).contiguous()
    draft = q.expand(rows, width, -1).contiguous()
    candidates = (
        torch.tensor([4, 0], device="cuda", dtype=torch.int32)
        .expand(rows, -1)
        .contiguous()
    )
    starts = torch.arange(rows + 1, device="cuda", dtype=torch.int64) * 2
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) * 97 + 3
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 53 + 11
    actual = _run_verify(api, (candidates, target, draft, starts, seeds, positions))
    rejected = actual[0].view(rows, width)[:, 0] != 0
    accept_rate = (~rejected).float().mean().item()
    assert abs(accept_rate - 0.25) < 0.025
    rejected_tokens = actual[0].view(rows, width)[rejected, 0].long()
    counts = torch.bincount(rejected_tokens, minlength=vocab).float()
    residual = torch.clamp(p - _RESIDUAL_PENALTY * q, min=0)
    residual_probs = residual / residual.sum()
    expected = rejected.sum() * residual_probs
    sigma = torch.sqrt(rejected.sum() * residual_probs * (1 - residual_probs))
    assert counts[0].item() == 0
    assert bool(((counts - expected).abs() <= 7 * sigma + 2).all())


@pytest.mark.parametrize(
    ("target_values", "draft_values"),
    [
        # q has no support on target tokens 2 and 3.
        ([0.05, 0.60, 0.25, 0.10], [0.65, 0.35, 0.0, 0.0]),
        # With matching distributions, every proposal is accepted.
        ([0.10, 0.20, 0.30, 0.40], [0.10, 0.20, 0.30, 0.40]),
        # A deterministic proposal forces the residual path for other tokens.
        ([0.20, 0.50, 0.30, 0.0], [1.0, 0.0, 0.0, 0.0]),
    ],
    ids=["target-mass-outside-draft-support", "matching", "deterministic-draft"],
)
def test_verify_packed_chain_first_token_tracks_target_law(
    target_values: list[float], draft_values: list[float]
) -> None:
    """Check the complete proposal and accept/reject path against target p.

    Verification uses max(p - exp(-5e-5) * q, 0) on rejection, so this is a
    statistical sanity check for that documented epsilon approximation, not a
    proof of an exact target distribution.
    """
    _need_cuda()
    api = _api()
    rows, width, vocab = 65536, 2, len(target_values)
    target_row = torch.tensor(target_values, device="cuda", dtype=torch.float32)
    draft_row = torch.tensor(draft_values, device="cuda", dtype=torch.float32)
    target = target_row.expand(rows * width, -1).contiguous()
    draft = draft_row.expand(rows, width, -1).contiguous()
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + 41003
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 53 + 17

    proposals = torch.empty(rows, device="cuda", dtype=torch.int32)
    api.sample_probs(
        probs=draft_row.expand(rows, -1).contiguous(),
        seeds=seeds,
        positions=positions,
        output=proposals,
        rng_tag=_DRAFT,
    )
    candidates = torch.zeros((rows, width), device="cuda", dtype=torch.int32)
    candidates[:, 1] = proposals
    query_start_loc = torch.arange(rows + 1, device="cuda", dtype=torch.int64) * width
    predicts, _, _ = _run_verify(
        api, (candidates, target, draft, query_start_loc, seeds, positions)
    )

    counts = torch.bincount(predicts.view(rows, width)[:, 0].long(), minlength=vocab)
    expected = rows * target_row
    # Six binomial standard deviations plus two samples gives a fixed,
    # conservative per-token tolerance for all three categorical cases.
    sigma = torch.sqrt(rows * target_row * (1 - target_row))
    tolerance = 6 * sigma + 2
    assert bool(((counts - expected).abs() <= tolerance).all()), (
        f"observed counts {counts.tolist()} differ from target counts "
        f"{expected.tolist()} by more than {tolerance.tolist()}"
    )


def test_verify_packed_chain_is_invariant_to_request_reordering() -> None:
    _need_cuda()
    api = _api()
    inputs = _make_chain_inputs(batch=6)
    actual = _run_verify(api, inputs)
    candidates, target, draft, starts, seeds, positions = inputs
    permutation = torch.tensor([4, 1, 5, 0, 3, 2], device="cuda")
    lengths = starts[1:] - starts[:-1]
    reordered_lengths = lengths[permutation]
    reordered_starts = torch.cat(
        [torch.zeros(1, device="cuda", dtype=torch.int64), reordered_lengths.cumsum(0)]
    )
    row_parts = [target[starts[row] : starts[row + 1]] for row in permutation.tolist()]
    reordered_target = torch.cat(row_parts + [target[-2:]], dim=0)
    reordered_inputs = (
        candidates[permutation],
        reordered_target,
        draft[permutation],
        reordered_starts,
        seeds[permutation],
        positions[permutation],
    )
    reordered = _run_verify(api, reordered_inputs)
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        reordered[0].view(6, -1)[inverse], actual[0].view(6, -1), atol=0, rtol=0
    )
    width = candidates.shape[1]
    output_cols = torch.arange(6, device="cuda", dtype=torch.int32)[:, None] * width
    reordered_cols = torch.where(reordered[1] >= 0, reordered[1] - output_cols, -1)
    actual_cols = torch.where(actual[1] >= 0, actual[1] - output_cols, -1)
    torch.testing.assert_close(reordered_cols[inverse], actual_cols, atol=0, rtol=0)
    torch.testing.assert_close(reordered[2][inverse], actual[2], atol=0, rtol=0)


def test_verify_packed_chain_replays_changed_rng_inputs() -> None:
    _need_cuda()
    api = _api()
    inputs = _make_chain_inputs(batch=4)
    outputs = _run_verify(api, inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_verify_into(api, inputs, outputs)
    for iteration in range(4):
        inputs[4].add_(1)
        inputs[5].add_(10007)
        new_counts = ([1, 2, 4, 3], [3, 1, 2, 4])[iteration % 2]
        inputs[3].copy_(
            torch.tensor(
                [0, *torch.tensor(new_counts).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int64,
            )
        )
        graph.replay()
        eager = _run_verify(api, inputs)
        for captured, reference in zip(outputs, eager, strict=True):
            torch.testing.assert_close(captured, reference, atol=0, rtol=0)
        expected = _oracle_verify(
            candidates=inputs[0].cpu(),
            target_probs=inputs[1].cpu(),
            draft_probs=inputs[2].cpu(),
            query_start_loc=inputs[3].cpu(),
            seeds=inputs[4].cpu(),
            positions=inputs[5].cpu(),
        )
        for got, ref in zip(eager, expected, strict=True):
            torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)


def test_sample_and_verify_reuse_compiled_cute_kernels() -> None:
    _need_cuda()
    api = _api()
    from utils import assert_no_cute_compile

    for batch in (1, 3):
        _run_verify(api, _make_chain_inputs(batch=batch))
    probs = torch.full((2, 8), 1 / 8, device="cuda", dtype=torch.float32)
    seeds = torch.arange(2, device="cuda", dtype=torch.int64)
    output = torch.empty(2, device="cuda", dtype=torch.int32)
    api.sample_probs(
        probs=probs, seeds=seeds, positions=seeds, output=output, rng_tag=_TARGET
    )
    with assert_no_cute_compile():
        for batch in (1, 3, 5):
            _run_verify(api, _make_chain_inputs(batch=batch))
        for rows in (2, 5, 9):
            row_probs = probs[:1].expand(rows, -1).contiguous()
            row_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
            api.sample_probs(
                probs=row_probs,
                seeds=row_ids,
                positions=row_ids,
                output=torch.empty(rows, device="cuda", dtype=torch.int32),
                rng_tag=_TARGET,
            )


@pytest.mark.parametrize("far_operand", ["target", "draft"])
@pytest.mark.parametrize("accept", [False, True])
def test_verify_packed_chain_addresses_rows_past_int32(
    far_operand: str, accept: bool
) -> None:
    _need_cuda()
    api = _api()
    free_bytes, _ = torch.cuda.mem_get_info()
    vocab, width = 16384, 2
    start = 2**31 // vocab + 1 if far_operand == "target" else 0
    slot = 2**31 // (width * vocab) + 1 if far_operand == "draft" else 0
    rows, slots = start + width, slot + 1
    assert max(start * vocab, slot * width * vocab) > 2**31
    required = (rows * vocab + slots * width * vocab) * 4
    if required > free_bytes * 0.75:
        pytest.skip("far-end probability tensor exceeds available CUDA memory")
    target = torch.empty((rows, vocab), device="cuda", dtype=torch.float32)
    target[start:].zero_()
    target[start, 1] = 0.25
    target[start, 7] = 0.75
    target[start + 1, 11] = 1.0
    draft = torch.empty((slots, width, vocab), device="cuda", dtype=torch.float32)
    draft[slot].zero_()
    draft[slot, 0, 1] = 1.0
    candidates = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    seed = next(
        seed
        for seed in range(10000)
        if (_philox_uniform(seed, 4, _ACCEPT) < 0.25) == accept
    )
    inputs = (
        candidates,
        target,
        draft,
        torch.tensor([start, rows], device="cuda", dtype=torch.int64),
        torch.tensor([seed], device="cuda", dtype=torch.int64),
        torch.tensor([4], device="cuda", dtype=torch.int64),
    )
    result = _run_verify(
        api,
        inputs,
        draft_pool_indices=torch.tensor([slot], device="cuda", dtype=torch.int64),
    )
    assert result[0].tolist() == ([1, 11] if accept else [7, 0])
    assert result[1].tolist() == ([[0, 1]] if accept else [[0, -1]])
    assert result[2].tolist() == [2 if accept else 1]


def test_sample_probs_addresses_rows_past_int32() -> None:
    _need_cuda()
    api = _api()
    vocab, rows = 16384, 131074
    assert (rows - 1) * vocab > 2**31
    free_bytes, _ = torch.cuda.mem_get_info()
    if rows * vocab * 4 > free_bytes * 0.75:
        pytest.skip("far-end probability tensor exceeds available CUDA memory")
    probs = torch.zeros((rows, vocab), device="cuda", dtype=torch.float32)
    probs[-1, 7] = 1.0
    output = torch.empty(rows, device="cuda", dtype=torch.int32)
    api.sample_probs(
        probs=probs,
        seeds=torch.zeros(rows, device="cuda", dtype=torch.int64),
        positions=torch.zeros(rows, device="cuda", dtype=torch.int64),
        output=output,
        rng_tag=_DRAFT,
    )
    assert output[0].item() == 0
    assert output[-1].item() == 7


@pytest.mark.parametrize(("proposal", "length"), [(2.0, 2), (2.5, 1)])
def test_verify_proposal_sentinel_requires_strictly_greater_probability(
    proposal: float, length: int
) -> None:
    _need_cuda()
    api = _api()
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device="cuda")
    draft = torch.zeros((1, 2, 2), device="cuda")
    draft[0, 0, 1] = proposal
    seed = next(
        seed for seed in range(10000) if _philox_uniform(seed, 0, _ACCEPT) < 0.5
    )
    result = _run_verify(
        api,
        (
            torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
            target,
            draft,
            torch.tensor([0, 2], device="cuda", dtype=torch.int64),
            torch.tensor([seed], device="cuda", dtype=torch.int64),
            torch.tensor([0], device="cuda", dtype=torch.int64),
        ),
    )
    assert result[2].tolist() == [length]


def test_verify_proposal_sentinel_cutoff_is_a_runtime_argument() -> None:
    _need_cuda()
    api = _api()
    from utils import assert_no_cute_compile

    probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").repeat(2, 1)
    draft = torch.full((1, 2, 4), 3.0, device="cuda")
    candidates = torch.zeros((1, 2), device="cuda", dtype=torch.int32)
    outputs = (
        torch.empty((2,), device="cuda", dtype=torch.int32),
        torch.empty((1, 2), device="cuda", dtype=torch.int32),
        torch.empty((1,), device="cuda", dtype=torch.int32),
    )

    def run(threshold):
        api.verify_packed_chain(
            predicts=outputs[0],
            accept_index=outputs[1],
            accept_length=outputs[2],
            candidates=candidates,
            target_probs=probabilities,
            draft_probs=draft,
            query_start_loc=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
            seeds=torch.tensor([1], device="cuda", dtype=torch.int64),
            positions=torch.tensor([123], device="cuda", dtype=torch.int64),
            reject_draft_prob_threshold=threshold,
        )
        assert outputs[2].item() == 1

    run(2.0)
    with assert_no_cute_compile():
        draft.fill_(2.0)
        run(1.0)
