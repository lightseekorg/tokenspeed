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

"""Shared chain verification preserves the complete greedy Torch outputs."""

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton import verify_chain_target_sampled
from tokenspeed_kernel.ops.sampling.triton.verify import (
    _verify_chain_target_sampled_kernel,
)
from utils import assert_no_triton_compile


def _compact_reference(candidates, compact_target, lengths):
    batch, width = candidates.shape
    tokens = torch.zeros(batch * width, dtype=torch.int32, device=candidates.device)
    indices = torch.full_like(candidates, -1)
    counts = torch.full((batch,), -1, dtype=torch.int32, device=candidates.device)
    start = 0
    for request, length in enumerate(lengths):
        if length:
            target = compact_target[start : start + length].to(torch.int32)
            tokens[request * width : request * width + length] = target
            accepted = 0
            while (
                accepted + 1 < length
                and candidates[request, accepted + 1] == target[accepted]
            ):
                accepted += 1
            counts[request] = accepted
            indices[request, : accepted + 1] = torch.arange(
                request * width,
                request * width + accepted + 1,
                device=candidates.device,
            )
        start += length
    return tokens, indices, counts


@pytest.mark.parametrize("width", [1, 4, 9])
@pytest.mark.parametrize("pattern", ["none", "all", "partial", "cast_wrap"])
def test_compact_greedy_chain_replay_reads_only_selected_prefixes(
    device, width, pattern
):
    batch = 4
    candidates, target = _inputs(batch, width, pattern, device)
    compact = torch.empty_like(target).flatten()
    offsets = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    actual = _outputs(batch, width, device)

    def load(lengths):
        prefix = [0]
        for length in lengths:
            prefix.append(prefix[-1] + length)
        compact.fill_(-555)
        if prefix[-1]:
            compact[: prefix[-1]].copy_(
                torch.cat([target[i, :n] for i, n in enumerate(lengths)])
            )
        offsets.copy_(torch.tensor(prefix, dtype=torch.int32, device=device))

    def run():
        verify_chain_target_sampled(
            *actual,
            candidates,
            compact,
            query_start_loc=offsets,
            write_rejected_targets=True,
            enable_pdl=False,
        )

    load([width] * batch)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    with assert_no_triton_compile(_verify_chain_target_sampled_kernel):
        for lengths in (
            [width, 1, 0, width],
            [0, width, 1, 0],
            [0] * batch,
            [width] * batch,
        ):
            load(lengths)
            graph.replay()
            for got, want in zip(
                actual, _compact_reference(candidates, compact, lengths)
            ):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_compact_greedy_chain_large_prediction_offsets(device):
    first = 2**31 + 1
    entries = first + 2
    if torch.cuda.mem_get_info(device)[0] < entries * 4 + 256 * 1024**2:
        pytest.skip("not enough memory for the sparse high-offset allocation")
    # Leave the allocation uninitialized and touch only its last two entries.
    target = torch.empty(entries, dtype=torch.int32, device=device)
    target[-2:].copy_(torch.tensor([17, 23], dtype=torch.int32, device=device))
    offsets = torch.tensor(
        [first, first + 1, first + 2], dtype=torch.int64, device=device
    )
    candidates = torch.zeros(2, 1, dtype=torch.int32, device=device)
    actual = _outputs(2, 1, device)
    verify_chain_target_sampled(
        *actual,
        candidates,
        target,
        query_start_loc=offsets,
        write_rejected_targets=True,
        enable_pdl=False,
    )
    torch.testing.assert_close(
        actual[0], torch.tensor([17, 23], dtype=torch.int32, device=device)
    )
    assert actual[2].tolist() == [0, 0]


def _inputs(bs, width, pattern, device):
    target = (
        torch.arange(bs * width, device=device, dtype=torch.int64).view(bs, width) + 100
    )
    candidates = torch.full((bs, width), -7, device=device, dtype=torch.int32)
    candidates[:, 1:] = target[:, :-1].to(torch.int32)
    if pattern != "all" and width > 1:
        accepted = (
            torch.arange(bs, device=device) % width
            if pattern == "partial"
            else torch.zeros(bs, device=device, dtype=torch.int64)
        )
        mismatch = accepted < width - 1
        row = torch.arange(bs, device=device)[mismatch]
        candidates[row, accepted[mismatch] + 1] = -2
    if pattern == "invalid_tail" and width > 1:
        target[:, -1] = -1
        candidates[:, -1] = -1
    if pattern == "cast_wrap" and width > 1:
        target[:, 0] += 2**32
        candidates[:, 1] = target[:, 0].to(torch.int32)
    return candidates, target


def _outputs(bs, width, device):
    return (
        torch.full((bs * width,), 777, device=device, dtype=torch.int32),
        torch.full((bs, width), 17, device=device, dtype=torch.int32),
        torch.full((bs,), 99, device=device, dtype=torch.int32),
    )


def _reference(candidates, target):
    # Keep this private oracle import in sync with runtime verifier refactors.
    from tokenspeed.runtime.sampling.backends.greedy import _verify_chain_greedy_torch

    bs, width = candidates.shape
    out = _outputs(bs, width, candidates.device)
    _verify_chain_greedy_torch(*out, candidates, target, bs, width)
    return out


@pytest.mark.parametrize("bs", [0, 1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("width", [1, 2, 6, 9, 32])
@pytest.mark.parametrize(
    "pattern", ["none", "all", "partial", "invalid_tail", "cast_wrap"]
)
def test_greedy_chain_full_outputs_match_torch(device, bs, width, pattern):
    candidates, target = _inputs(bs, width, pattern, device)
    expected = _reference(candidates, target)
    actual = _outputs(bs, width, device)
    verify_chain_target_sampled(
        *actual, candidates, target, write_rejected_targets=True, enable_pdl=False
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_sampled_chain_keeps_rejected_tails(device):
    candidates, target = _inputs(8, 6, "partial", device)
    full = _reference(candidates, target)
    actual = _outputs(8, 6, device)
    actual[1].fill_(-1)
    verify_chain_target_sampled(
        *actual, candidates, target, write_rejected_targets=False, enable_pdl=False
    )
    torch.testing.assert_close(actual[1], full[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], full[2], rtol=0, atol=0)
    columns = torch.arange(6, device=device)[None, :]
    expected_tokens = torch.where(columns <= full[2][:, None], full[0].view(8, 6), 777)
    torch.testing.assert_close(actual[0].view(8, 6), expected_tokens, rtol=0, atol=0)


def test_greedy_chain_graph_replay_refreshes_full_window(device):
    candidates, target = _inputs(8, 6, "all", device)
    actual = _outputs(8, 6, device)
    verify_chain_target_sampled(
        *actual, candidates, target, write_rejected_targets=True, enable_pdl=False
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify_chain_target_sampled(
            *actual, candidates, target, write_rejected_targets=True, enable_pdl=False
        )
    for pattern in ("all", "none", "partial", "invalid_tail"):
        new_candidates, new_target = _inputs(8, 6, pattern, device)
        candidates.copy_(new_candidates)
        target.copy_(new_target)
        graph.replay()
        for got, want in zip(actual, _reference(candidates, target)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_greedy_chain_no_row_or_stride_specialization(device):
    candidates, target = _inputs(32, 6, "partial", device)
    actual = _outputs(32, 6, device)
    verify_chain_target_sampled(
        *actual, candidates, target, write_rejected_targets=True, enable_pdl=False
    )
    with assert_no_triton_compile(_verify_chain_target_sampled_kernel):
        for bs in range(1, 33):
            verify_chain_target_sampled(
                actual[0][: bs * 6],
                actual[1][:bs],
                actual[2][:bs],
                candidates[:bs],
                target[:bs],
                write_rejected_targets=True,
                enable_pdl=False,
            )
        for padding in (1, 7, 31):
            storage = torch.empty(8, 6 + padding, dtype=torch.int32, device=device)
            strided_candidates = storage[:, :6].copy_(candidates[:8])
            out = (
                torch.empty(8 * 6 * 2, dtype=torch.int32, device=device)[::2],
                torch.empty(8, 6 + padding, dtype=torch.int32, device=device)[:, :6],
                torch.empty(16, dtype=torch.int32, device=device)[::2],
            )
            verify_chain_target_sampled(
                *out,
                strided_candidates,
                target[:8],
                write_rejected_targets=True,
                enable_pdl=False,
            )
            for got, want in zip(out, _reference(strided_candidates, target[:8])):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
