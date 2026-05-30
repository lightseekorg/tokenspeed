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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton import (
    _QRITA_PERCENTILE_TO_STD_TABLE,
    gumbel_sample_from_pools,
    gumbel_sample_from_pools_compact,
    gumbel_sample_from_pools_generic,
    gumbel_sample_min_p_from_pools,
    gumbel_sample_min_p_from_pools_parallel,
    gumbel_sample_top_k_top_p_from_pools,
    gumbel_sample_top_k_top_p_qrita_from_pools,
    gumbel_sample_top_p_parallel_from_pools,
)

# Sentinel matching tokenspeed.runtime.sampling.sampling_params._TOP_K_DISABLED.
_TOP_K_DISABLED = 1 << 30


def _gumbel_scratch(rows: int, vocab_size: int, device: str):
    num_blocks = (vocab_size + 1023) // 1024
    local_ids = torch.empty((rows, num_blocks), dtype=torch.int32, device=device)
    local_scores = torch.empty((rows, num_blocks), dtype=torch.float32, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)
    return local_ids, local_scores, out


def _top_k_top_p_gumbel_scratch(rows: int, vocab_size: int, device: str):
    num_blocks = (vocab_size + 2047) // 2048
    num_candidates = num_blocks * 128
    candidate_ids = torch.empty(
        (rows, num_candidates), dtype=torch.int32, device=device
    )
    candidate_logits = torch.empty(
        (rows, num_candidates), dtype=torch.float32, device=device
    )
    out = torch.empty((rows,), dtype=torch.int32, device=device)
    return candidate_ids, candidate_logits, out


def _top_p_parallel_scratch(
    rows: int, vocab_size: int, block_size: int, attempts: int, device: str
):
    num_blocks = (vocab_size + block_size - 1) // block_size
    local_max = torch.empty((rows, num_blocks), dtype=torch.float32, device=device)
    local_sum = torch.empty_like(local_max)
    local_argmax = torch.empty((rows, num_blocks), dtype=torch.int32, device=device)
    local_scores = torch.empty(
        (rows, num_blocks, attempts), dtype=torch.float32, device=device
    )
    local_logits = torch.empty_like(local_scores)
    local_ids = torch.empty(
        (rows, num_blocks, attempts), dtype=torch.int32, device=device
    )
    row_max = torch.empty((rows,), dtype=torch.float32, device=device)
    row_total = torch.empty_like(row_max)
    row_argmax = torch.empty((rows,), dtype=torch.int32, device=device)
    row_candidate_logits = torch.empty(
        (rows, attempts), dtype=torch.float32, device=device
    )
    row_candidate_ids = torch.empty((rows, attempts), dtype=torch.int32, device=device)
    accepted = torch.empty((rows,), dtype=torch.int32, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)
    return (
        local_max,
        local_sum,
        local_argmax,
        local_scores,
        local_logits,
        local_ids,
        row_max,
        row_total,
        row_argmax,
        row_candidate_logits,
        row_candidate_ids,
        accepted,
        out,
    )


def _qrita_gumbel_scratch(rows: int, vocab_size: int, device: str):
    buffer = torch.empty((rows, vocab_size), dtype=torch.float32, device=device)
    table = torch.tensor(
        _QRITA_PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=device
    )
    out = torch.empty((rows,), dtype=torch.int32, device=device)
    return buffer, table, out


def _top_k_topp_allowed_ids(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> set[int]:
    top_k = min(top_k, logits.numel())
    top_vals, top_ids = torch.topk(logits.float(), top_k, sorted=True)
    probs = torch.softmax(top_vals / temperature, dim=-1)
    cumulative_before = torch.cumsum(probs, dim=-1) - probs
    keep = cumulative_before < top_p
    return set(top_ids[keep].int().cpu().tolist())


def _top_k_topp_minp_allowed_ids(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, min_p: float
) -> set[int]:
    top_k = min(top_k, logits.numel())
    top_vals, top_ids = torch.topk(logits.float(), top_k, sorted=True)
    scaled = top_vals / temperature
    probs = torch.softmax(scaled, dim=-1)
    cumulative_before = torch.cumsum(probs, dim=-1) - probs
    keep = (cumulative_before < top_p) & (
        scaled >= scaled.max() + torch.log(torch.tensor(min_p, device=logits.device))
    )
    return set(top_ids[keep].int().cpu().tolist())


def test_gumbel_sample_from_pools_takes_tail_token(device: str) -> None:
    rows, vocab_size = 3, 1025
    logits = torch.full((rows, vocab_size), -10.0, dtype=torch.float32, device=device)
    logits[0, vocab_size - 1] = 1.0e6
    logits[1, vocab_size - 17] = 1.0e6
    logits[2, vocab_size - 33] = 1.0e6
    req_pool_indices = torch.tensor([4, 2, 7], dtype=torch.int32, device=device)
    pool_rows = 9
    temperature_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    seed_pool = torch.arange(123, 123 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)
    local_ids, local_scores, out = _gumbel_scratch(rows, vocab_size, device)

    sampled = gumbel_sample_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        out,
    )

    torch.testing.assert_close(
        sampled.cpu(),
        torch.tensor(
            [vocab_size - 1, vocab_size - 17, vocab_size - 33], dtype=torch.int32
        ),
    )


def test_gumbel_sample_from_pools_compact_matches_regular_pool_path(
    device: str,
) -> None:
    torch.manual_seed(112)
    rows, vocab_size = 3, 2049
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device)
    req_pool_indices = torch.tensor([4, 2, 7], dtype=torch.int32, device=device)
    pool_rows = 9
    temperature_pool = torch.linspace(
        0.5, 1.5, pool_rows, dtype=torch.float32, device=device
    )
    seed_pool = torch.arange(123, 123 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)
    local_ids, local_scores, regular_out = _gumbel_scratch(rows, vocab_size, device)
    compact_out = torch.empty((rows,), dtype=torch.int32, device=device)

    regular = gumbel_sample_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        regular_out,
    ).clone()
    compact = gumbel_sample_from_pools_compact(
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        compact_out,
        block_size=1024,
    ).clone()

    torch.testing.assert_close(compact, regular)


@pytest.mark.parametrize("use_compact", [True, False])
def test_gumbel_no_filter_verify_idx_mapping_matches_expanded_rows(
    device: str, use_compact: bool
) -> None:
    torch.manual_seed(2027)
    bs, n = 2, 3
    rows = bs * n
    vocab_size = 1025 if use_compact else 4097
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.8, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    seed_pool = torch.arange(701, 701 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(19, 19 + pool_rows, dtype=torch.int64, device=device)

    mapped_out = torch.empty((rows,), dtype=torch.int32, device=device)
    if use_compact:
        mapped = gumbel_sample_from_pools_compact(
            logits,
            req_pool_indices,
            temperature_pool,
            seed_pool,
            offsets_pool,
            mapped_out,
            block_size=1024,
            num_tokens_per_req=n,
        ).clone()
    else:
        local_ids, local_scores, _ = _gumbel_scratch(rows, vocab_size, device)
        mapped = gumbel_sample_from_pools(
            logits,
            req_pool_indices,
            temperature_pool,
            seed_pool,
            offsets_pool,
            local_ids,
            local_scores,
            mapped_out,
            num_tokens_per_req=n,
        ).clone()

    expanded_req = torch.arange(rows, dtype=torch.int32, device=device)
    expanded_temperature = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_seed = torch.empty((rows,), dtype=torch.int64, device=device)
    expanded_offsets = torch.empty((rows,), dtype=torch.int64, device=device)
    for row in range(rows):
        pool_idx = int(req_pool_indices[row // n].item())
        expanded_temperature[row] = temperature_pool[pool_idx]
        expanded_seed[row] = seed_pool[pool_idx]
        expanded_offsets[row] = offsets_pool[pool_idx] + row % n

    expanded_out = torch.empty((rows,), dtype=torch.int32, device=device)
    if use_compact:
        expanded = gumbel_sample_from_pools_compact(
            logits,
            expanded_req,
            expanded_temperature,
            expanded_seed,
            expanded_offsets,
            expanded_out,
            block_size=1024,
        ).clone()
    else:
        local_ids, local_scores, _ = _gumbel_scratch(rows, vocab_size, device)
        expanded = gumbel_sample_from_pools(
            logits,
            expanded_req,
            expanded_temperature,
            expanded_seed,
            expanded_offsets,
            local_ids,
            local_scores,
            expanded_out,
        ).clone()

    torch.testing.assert_close(mapped, expanded)


def test_gumbel_generic_mixed_batch_samples_allowed_set(device: str) -> None:
    torch.manual_seed(321)
    rows, vocab_size = 4, 257
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.5
    req_pool_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.8, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.tensor(
        [1, _TOP_K_DISABLED, 16, _TOP_K_DISABLED, 32],
        dtype=torch.int32,
        device=device,
    )
    top_p_pool = torch.tensor(
        [1.0, 1.0, 0.9, 0.75, 0.85],
        dtype=torch.float32,
        device=device,
    )
    seed_pool = torch.arange(99, 99 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(7, 7 + pool_rows, dtype=torch.int64, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)

    first = gumbel_sample_from_pools_generic(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        out,
    ).clone()
    second = gumbel_sample_from_pools_generic(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        out,
    ).clone()

    torch.testing.assert_close(first, second)
    for row, token_id in enumerate(first.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        top_k = int(top_k_pool[pool_idx].item())
        if top_k == _TOP_K_DISABLED:
            top_k = logits.shape[1]
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            top_k,
            float(top_p_pool[pool_idx].item()),
        )
        assert token_id in allowed


def test_gumbel_generic_verify_idx_mapping_matches_expanded_rows(device: str) -> None:
    torch.manual_seed(2026)
    bs, n, vocab_size = 2, 3, 257
    rows = bs * n
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.8, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.tensor(
        [1, _TOP_K_DISABLED, 32, _TOP_K_DISABLED, 16],
        dtype=torch.int32,
        device=device,
    )
    top_p_pool = torch.tensor(
        [1.0, 0.85, 0.9, 0.75, 0.95], dtype=torch.float32, device=device
    )
    seed_pool = torch.arange(101, 101 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)

    mapped_out = torch.empty((rows,), dtype=torch.int32, device=device)
    mapped = gumbel_sample_from_pools_generic(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        mapped_out,
        num_tokens_per_req=n,
    ).clone()

    expanded_req = torch.arange(rows, dtype=torch.int32, device=device)
    expanded_temperature = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_top_k = torch.empty((rows,), dtype=torch.int32, device=device)
    expanded_top_p = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_seed = torch.empty((rows,), dtype=torch.int64, device=device)
    expanded_offsets = torch.empty((rows,), dtype=torch.int64, device=device)
    for row in range(rows):
        pool_idx = int(req_pool_indices[row // n].item())
        expanded_temperature[row] = temperature_pool[pool_idx]
        expanded_top_k[row] = top_k_pool[pool_idx]
        expanded_top_p[row] = top_p_pool[pool_idx]
        expanded_seed[row] = seed_pool[pool_idx]
        expanded_offsets[row] = offsets_pool[pool_idx] + row % n

    expanded_out = torch.empty((rows,), dtype=torch.int32, device=device)
    expanded = gumbel_sample_from_pools_generic(
        logits,
        expanded_req,
        expanded_temperature,
        expanded_top_k,
        expanded_top_p,
        expanded_seed,
        expanded_offsets,
        expanded_out,
    ).clone()

    torch.testing.assert_close(mapped, expanded)


def test_gumbel_top_k_equals_argmax_for_k1(device: str) -> None:
    torch.manual_seed(12)
    rows, vocab_size = 4, 257
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device)
    req_pool_indices = torch.tensor([4, 2, 7, 1], dtype=torch.int32, device=device)
    pool_rows = 9
    temperature_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    top_k_pool = torch.ones((pool_rows,), dtype=torch.int32, device=device)
    top_p_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    seed_pool = torch.arange(123, 123 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)
    candidate_ids, candidate_logits, out = _top_k_top_p_gumbel_scratch(
        rows, vocab_size, device
    )

    sampled = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
    )

    torch.testing.assert_close(sampled, torch.argmax(logits, dim=-1).to(torch.int32))


@pytest.mark.parametrize("top_k,top_p", [(8, 1.0), (16, 0.75), (64, 0.9)])
def test_gumbel_top_k_top_p_samples_allowed_set(
    device: str, top_k: int, top_p: float
) -> None:
    torch.manual_seed(42 + top_k)
    rows, vocab_size = 3, 513
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 3.0
    # Mask a very tempting token to prove -inf logits stay out of the sample set.
    logits[:, vocab_size - 1] = float("-inf")
    req_pool_indices = torch.tensor([4, 2, 7], dtype=torch.int32, device=device)
    pool_rows = 9
    temperature_pool = torch.linspace(
        0.7, 1.3, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.full((pool_rows,), top_k, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), top_p, dtype=torch.float32, device=device)
    seed_pool = torch.arange(123, 123 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)
    candidate_ids, candidate_logits, out = _top_k_top_p_gumbel_scratch(
        rows, vocab_size, device
    )

    sampled = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            top_k,
            top_p,
        )
        assert token_id in allowed


def test_gumbel_top_k_top_p_min_p_samples_allowed_set(device: str) -> None:
    torch.manual_seed(43)
    rows, vocab_size = 3, 513
    top_k, top_p, min_p = 64, 0.9, 0.15
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 3.0
    req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    pool_rows = 4
    temperature_pool = torch.linspace(
        0.7, 1.1, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.full((pool_rows,), top_k, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), top_p, dtype=torch.float32, device=device)
    min_p_pool = torch.full((pool_rows,), min_p, dtype=torch.float32, device=device)
    seed_pool = torch.arange(321, 321 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(13, 13 + pool_rows, dtype=torch.int64, device=device)
    candidate_ids, candidate_logits, out = _top_k_top_p_gumbel_scratch(
        rows, vocab_size, device
    )

    sampled = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
        min_p_pool=min_p_pool,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_minp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            top_k,
            top_p,
            min_p,
        )
        assert token_id in allowed


def test_gumbel_top_k_top_p_is_deterministic(device: str) -> None:
    torch.manual_seed(99)
    rows, vocab_size = 2, 1025
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device)
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    top_k_pool = torch.full((pool_rows,), 32, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), 0.9, dtype=torch.float32, device=device)
    seed_pool = torch.arange(11, 11 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(5, 5 + pool_rows, dtype=torch.int64, device=device)
    candidate_ids, candidate_logits, out = _top_k_top_p_gumbel_scratch(
        rows, vocab_size, device
    )

    first = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
    ).clone()
    second = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
    ).clone()

    torch.testing.assert_close(first, second)


def test_gumbel_top_k_top_p_large_vocab_samples_allowed_set(device: str) -> None:
    torch.manual_seed(2029)
    rows, vocab_size = 2, 32768
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    logits[:, -1] = float("-inf")
    req_pool_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    pool_rows = 3
    temperature_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    top_k_pool = torch.full((pool_rows,), 128, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), 0.9, dtype=torch.float32, device=device)
    seed_pool = torch.arange(31, 31 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(9, 9 + pool_rows, dtype=torch.int64, device=device)
    candidate_ids, candidate_logits, out = _top_k_top_p_gumbel_scratch(
        rows, vocab_size, device
    )

    sampled = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            int(top_k_pool[pool_idx].item()),
            float(top_p_pool[pool_idx].item()),
        )
        assert token_id in allowed


@pytest.mark.parametrize("top_k,top_p", [(1, 1.0), (16, 0.75), (64, 0.9)])
def test_gumbel_top_k_top_p_qrita_samples_allowed_set(
    device: str, top_k: int, top_p: float
) -> None:
    torch.manual_seed(5020 + top_k)
    rows, vocab_size = 3, 513
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    logits[:, vocab_size - 1] = float("-inf")
    req_pool_indices = torch.tensor([4, 2, 7], dtype=torch.int32, device=device)
    pool_rows = 9
    temperature_pool = torch.linspace(
        0.7, 1.3, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.full((pool_rows,), top_k, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), top_p, dtype=torch.float32, device=device)
    seed_pool = torch.arange(123, 123 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)
    qrita_buffer, qrita_table, out = _qrita_gumbel_scratch(rows, vocab_size, device)

    sampled = gumbel_sample_top_k_top_p_qrita_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        qrita_buffer,
        qrita_table,
        out,
        num_programs=rows,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            top_k,
            top_p,
        )
        assert token_id in allowed


def test_gumbel_top_k_top_p_qrita_verify_idx_mapping_matches_expanded_rows(
    device: str,
) -> None:
    torch.manual_seed(5030)
    bs, n = 2, 3
    rows, vocab_size = bs * n, 1025
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.8, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.full((pool_rows,), 64, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), 0.85, dtype=torch.float32, device=device)
    seed_pool = torch.arange(701, 701 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(19, 19 + pool_rows, dtype=torch.int64, device=device)

    mapped_buffer, mapped_table, mapped_out = _qrita_gumbel_scratch(
        rows, vocab_size, device
    )
    mapped = gumbel_sample_top_k_top_p_qrita_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        mapped_buffer,
        mapped_table,
        mapped_out,
        num_tokens_per_req=n,
        num_programs=rows,
    ).clone()

    expanded_req = torch.arange(rows, dtype=torch.int32, device=device)
    expanded_temperature = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_top_k = torch.empty((rows,), dtype=torch.int32, device=device)
    expanded_top_p = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_seed = torch.empty((rows,), dtype=torch.int64, device=device)
    expanded_offsets = torch.empty((rows,), dtype=torch.int64, device=device)
    for row in range(rows):
        pool_idx = int(req_pool_indices[row // n].item())
        expanded_temperature[row] = temperature_pool[pool_idx]
        expanded_top_k[row] = top_k_pool[pool_idx]
        expanded_top_p[row] = top_p_pool[pool_idx]
        expanded_seed[row] = seed_pool[pool_idx]
        expanded_offsets[row] = offsets_pool[pool_idx] + row % n

    expanded_buffer, expanded_table, expanded_out = _qrita_gumbel_scratch(
        rows, vocab_size, device
    )
    expanded = gumbel_sample_top_k_top_p_qrita_from_pools(
        logits,
        expanded_req,
        expanded_temperature,
        expanded_top_k,
        expanded_top_p,
        expanded_seed,
        expanded_offsets,
        expanded_buffer,
        expanded_table,
        expanded_out,
        num_programs=rows,
    ).clone()

    torch.testing.assert_close(mapped, expanded)


def test_gumbel_top_k_top_p_verify_idx_mapping_matches_expanded_rows(
    device: str,
) -> None:
    torch.manual_seed(2030)
    bs, n = 2, 3
    rows, vocab_size = bs * n, 1025
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.8, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.full((pool_rows,), 64, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), 0.85, dtype=torch.float32, device=device)
    seed_pool = torch.arange(701, 701 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(19, 19 + pool_rows, dtype=torch.int64, device=device)

    mapped_candidate_ids, mapped_candidate_logits, mapped_out = (
        _top_k_top_p_gumbel_scratch(rows, vocab_size, device)
    )
    mapped = gumbel_sample_top_k_top_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        mapped_candidate_ids,
        mapped_candidate_logits,
        mapped_out,
        num_tokens_per_req=n,
    ).clone()

    expanded_req = torch.arange(rows, dtype=torch.int32, device=device)
    expanded_temperature = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_top_k = torch.empty((rows,), dtype=torch.int32, device=device)
    expanded_top_p = torch.empty((rows,), dtype=torch.float32, device=device)
    expanded_seed = torch.empty((rows,), dtype=torch.int64, device=device)
    expanded_offsets = torch.empty((rows,), dtype=torch.int64, device=device)
    for row in range(rows):
        pool_idx = int(req_pool_indices[row // n].item())
        expanded_temperature[row] = temperature_pool[pool_idx]
        expanded_top_k[row] = top_k_pool[pool_idx]
        expanded_top_p[row] = top_p_pool[pool_idx]
        expanded_seed[row] = seed_pool[pool_idx]
        expanded_offsets[row] = offsets_pool[pool_idx] + row % n

    expanded_candidate_ids, expanded_candidate_logits, expanded_out = (
        _top_k_top_p_gumbel_scratch(rows, vocab_size, device)
    )
    expanded = gumbel_sample_top_k_top_p_from_pools(
        logits,
        expanded_req,
        expanded_temperature,
        expanded_top_k,
        expanded_top_p,
        expanded_seed,
        expanded_offsets,
        expanded_candidate_ids,
        expanded_candidate_logits,
        expanded_out,
    ).clone()

    torch.testing.assert_close(mapped, expanded)


def test_gumbel_generic_top_p_only_samples_allowed_set(device: str) -> None:
    torch.manual_seed(123)
    rows, vocab_size = 3, 513
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 3.0
    logits[:, vocab_size - 1] = float("-inf")
    req_pool_indices = torch.tensor([4, 2, 7], dtype=torch.int32, device=device)
    pool_rows = 9
    temperature_pool = torch.linspace(
        0.7, 1.3, pool_rows, dtype=torch.float32, device=device
    )
    top_k_pool = torch.full(
        (pool_rows,), _TOP_K_DISABLED, dtype=torch.int32, device=device
    )
    top_p_pool = torch.full((pool_rows,), 0.8, dtype=torch.float32, device=device)
    seed_pool = torch.arange(123, 123 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(17, 17 + pool_rows, dtype=torch.int64, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)

    sampled = gumbel_sample_from_pools_generic(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        out,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            logits.shape[1],
            float(top_p_pool[pool_idx].item()),
        )
        assert token_id in allowed


def test_gumbel_top_p_parallel_samples_allowed_set(device: str) -> None:
    torch.manual_seed(456)
    rows, vocab_size = 5, 4097
    block_size, attempts = 512, 3
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    logits[:, -1] = float("-inf")
    req_pool_indices = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
    pool_rows = 6
    temperature_pool = torch.linspace(
        0.7, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    top_p_pool = torch.full((pool_rows,), 0.8, dtype=torch.float32, device=device)
    seed_pool = torch.arange(333, 333 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(11, 11 + pool_rows, dtype=torch.int64, device=device)
    scratch = _top_p_parallel_scratch(rows, vocab_size, block_size, attempts, device)

    sampled = gumbel_sample_top_p_parallel_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        *scratch,
        block_size=block_size,
        num_attempts=attempts,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            vocab_size,
            float(top_p_pool[pool_idx].item()),
        )
        assert token_id in allowed


def test_gumbel_top_p_parallel_repair_is_deterministic_and_allowed(
    device: str,
) -> None:
    torch.manual_seed(789)
    rows, vocab_size = 6, 769
    block_size, attempts = 128, 1
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.5
    req_pool_indices = torch.tensor(
        [1, 2, 3, 4, 5, 6], dtype=torch.int32, device=device
    )
    pool_rows = 7
    temperature_pool = torch.linspace(
        0.8, 1.1, pool_rows, dtype=torch.float32, device=device
    )
    top_p_pool = torch.full((pool_rows,), 0.25, dtype=torch.float32, device=device)
    seed_pool = torch.arange(444, 444 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(5, 5 + pool_rows, dtype=torch.int64, device=device)
    first_scratch = _top_p_parallel_scratch(
        rows, vocab_size, block_size, attempts, device
    )
    second_scratch = _top_p_parallel_scratch(
        rows, vocab_size, block_size, attempts, device
    )

    first = gumbel_sample_top_p_parallel_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        *first_scratch,
        block_size=block_size,
        num_attempts=attempts,
    )
    second = gumbel_sample_top_p_parallel_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        *second_scratch,
        block_size=block_size,
        num_attempts=attempts,
    )

    torch.testing.assert_close(first, second)

    for row, token_id in enumerate(first.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        allowed = _top_k_topp_allowed_ids(
            logits[row],
            float(temperature_pool[pool_idx].item()),
            vocab_size,
            float(top_p_pool[pool_idx].item()),
        )
        assert token_id in allowed


def test_gumbel_top_p_parallel_verify_idx_mapping_matches_expanded_rows(
    device: str,
) -> None:
    torch.manual_seed(987)
    bs, n, vocab_size = 2, 3, 521
    rows = bs * n
    block_size, attempts = 128, 3
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([2, 4], dtype=torch.int32, device=device)
    pool_rows = 6
    temperature_pool = torch.linspace(
        0.75, 1.25, pool_rows, dtype=torch.float32, device=device
    )
    top_p_pool = torch.full((pool_rows,), 0.8, dtype=torch.float32, device=device)
    seed_pool = torch.arange(700, 700 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(10, 10 + pool_rows, dtype=torch.int64, device=device)
    mapped_scratch = _top_p_parallel_scratch(
        rows, vocab_size, block_size, attempts, device
    )
    mapped = gumbel_sample_top_p_parallel_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        *mapped_scratch,
        block_size=block_size,
        num_attempts=attempts,
        num_tokens_per_req=n,
    ).clone()

    expanded_req_pool_indices = torch.arange(
        1, rows + 1, dtype=torch.int32, device=device
    )
    expanded_temperature = torch.empty((rows + 1,), dtype=torch.float32, device=device)
    expanded_top_p = torch.empty((rows + 1,), dtype=torch.float32, device=device)
    expanded_seed = torch.empty((rows + 1,), dtype=torch.int64, device=device)
    expanded_offsets = torch.empty((rows + 1,), dtype=torch.int64, device=device)
    for row in range(rows):
        req_row = row // n
        spec_pos = row - req_row * n
        src_pool = int(req_pool_indices[req_row].item())
        dst_pool = row + 1
        expanded_temperature[dst_pool] = temperature_pool[src_pool]
        expanded_top_p[dst_pool] = top_p_pool[src_pool]
        expanded_seed[dst_pool] = seed_pool[src_pool]
        expanded_offsets[dst_pool] = offsets_pool[src_pool] + spec_pos
    expanded_scratch = _top_p_parallel_scratch(
        rows, vocab_size, block_size, attempts, device
    )
    expanded = gumbel_sample_top_p_parallel_from_pools(
        logits,
        expanded_req_pool_indices,
        expanded_temperature,
        expanded_top_p,
        expanded_seed,
        expanded_offsets,
        *expanded_scratch,
        block_size=block_size,
        num_attempts=attempts,
    ).clone()

    torch.testing.assert_close(mapped, expanded)


def test_gumbel_generic_min_p_samples_allowed_set(device: str) -> None:
    torch.manual_seed(777)
    rows, vocab_size = 3, 257
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    pool_rows = 4
    temperature_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    top_k_pool = torch.full(
        (pool_rows,), _TOP_K_DISABLED, dtype=torch.int32, device=device
    )
    top_p_pool = torch.ones((pool_rows,), dtype=torch.float32, device=device)
    min_p_pool = torch.full((pool_rows,), 0.2, dtype=torch.float32, device=device)
    seed_pool = torch.arange(31, 31 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(9, 9 + pool_rows, dtype=torch.int64, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)

    sampled = gumbel_sample_from_pools_generic(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        out,
        min_p_pool=min_p_pool,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        scaled = logits[row].float() / float(temperature_pool[pool_idx].item())
        probs = torch.softmax(scaled, dim=-1)
        threshold = float(min_p_pool[pool_idx].item()) * probs.max()
        assert probs[token_id] >= threshold


def test_gumbel_min_p_samples_allowed_set(device: str) -> None:
    torch.manual_seed(778)
    rows, vocab_size = 4, 1025
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.7, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    min_p_pool = torch.full((pool_rows,), 0.2, dtype=torch.float32, device=device)
    seed_pool = torch.arange(41, 41 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(3, 3 + pool_rows, dtype=torch.int64, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)

    first = gumbel_sample_min_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        out,
        block_size=256,
    ).clone()
    second = gumbel_sample_min_p_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        out,
        block_size=256,
    ).clone()

    torch.testing.assert_close(first, second)
    for row, token_id in enumerate(first.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        scaled = logits[row].float() / float(temperature_pool[pool_idx].item())
        probs = torch.softmax(scaled, dim=-1)
        threshold = float(min_p_pool[pool_idx].item()) * probs.max()
        assert probs[token_id] >= threshold


def test_gumbel_min_p_parallel_matches_allowed_set(device: str) -> None:
    torch.manual_seed(779)
    rows, vocab_size = 4, 4097
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    req_pool_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    pool_rows = 5
    temperature_pool = torch.linspace(
        0.7, 1.2, pool_rows, dtype=torch.float32, device=device
    )
    min_p_pool = torch.full((pool_rows,), 0.2, dtype=torch.float32, device=device)
    seed_pool = torch.arange(51, 51 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.arange(7, 7 + pool_rows, dtype=torch.int64, device=device)
    block_size = 512
    num_blocks = (vocab_size + block_size - 1) // block_size
    local_ids = torch.empty((rows, num_blocks), dtype=torch.int32, device=device)
    local_scores = torch.empty((rows, num_blocks), dtype=torch.float32, device=device)
    row_max = torch.empty((rows,), dtype=torch.float32, device=device)
    out = torch.empty((rows,), dtype=torch.int32, device=device)

    first = gumbel_sample_min_p_from_pools_parallel(
        logits,
        req_pool_indices,
        temperature_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        row_max,
        out,
        block_size=block_size,
    ).clone()
    second = gumbel_sample_min_p_from_pools_parallel(
        logits,
        req_pool_indices,
        temperature_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        row_max,
        out,
        block_size=block_size,
    ).clone()

    torch.testing.assert_close(first, second)
    for row, token_id in enumerate(first.cpu().tolist()):
        pool_idx = int(req_pool_indices[row].item())
        scaled = logits[row].float() / float(temperature_pool[pool_idx].item())
        probs = torch.softmax(scaled, dim=-1)
        threshold = float(min_p_pool[pool_idx].item()) * probs.max()
        assert probs[token_id] >= threshold
