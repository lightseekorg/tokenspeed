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

"""Bounded MLA history plans, page indices, and attention partition accuracy."""

import importlib.util
import itertools
import os
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
    chunking,
    create_chunked_cache_kv_indices_paged,
    plan_prefix_chunks,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def test_history_capacity_is_reused_without_skipping_or_repeating_rows():
    rng = random.Random(71)
    cases = [
        ([], 32768),
        ([0, 0], 32768),
        ([186219, 0], 32768),
        ([35422, 34576, 253783, 54497], 32768),
    ]
    for _ in range(200):
        batch = rng.randrange(1, 17)
        cases.append(
            (
                [rng.randrange(0, 30000) for _ in range(batch)],
                batch * 1024 + rng.randrange(1024),
            )
        )
    for lengths, capacity in cases:
        chunks = plan_prefix_chunks(torch.tensor(lengths, dtype=torch.int32), capacity)
        assert chunks.starts.shape == (-(-sum(lengths) // capacity), len(lengths))
        assert chunks.boundaries.shape == (chunks.starts.shape[0] + 1, len(lengths))
        assert not chunks.boundaries[0].any()
        torch.testing.assert_close(
            chunks.boundaries[-1], torch.tensor(lengths, dtype=torch.int32)
        )
        if not chunks.starts.shape[0]:
            continue
        torch.testing.assert_close(chunks.starts[1:], chunks.ends[:-1])
        assert bool((chunks.len_in_chunk >= 0).all())
        torch.testing.assert_close(
            chunks.len_in_chunk.sum(0), torch.tensor(lengths, dtype=torch.int64)
        )
        assert bool((chunks.len_in_chunk.sum(1)[:-1] == capacity).all())
        assert chunks.len_in_chunk.sum(1)[-1] <= capacity


@pytest.mark.parametrize(
    "lengths,capacity", [([1, 0, 63], 64), ([65536] * 4, 32768), ([186219], 32768)]
)
def test_small_partitions_and_uniform_prefixes_keep_existing_boundaries(
    lengths, capacity
):
    width = capacity // len(lengths)
    lengths = torch.tensor(lengths)
    old = chunking(lengths, -(-int(lengths.max()) // width), len(lengths), width)
    new = plan_prefix_chunks(lengths, capacity)
    torch.testing.assert_close(new.boundaries, old.boundaries)
    torch.testing.assert_close(new.len_in_chunk, old.len_in_chunk)


@pytest.mark.parametrize("lengths,capacity", [([-1], 4096), ([1, 2], 1), ([1], 0)])
def test_invalid_history_budget_or_lengths(lengths, capacity):
    with pytest.raises(ValueError):
        plan_prefix_chunks(torch.tensor(lengths), capacity)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("page_size", [1, 64])
def test_page_indices_and_graph_replay(monkeypatch, page_size):
    lengths = torch.tensor([12345, 0, 37, 23456], dtype=torch.int32)
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8192)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 4)
    columns = -(-int(lengths.max()) // page_size)
    table = (
        torch.arange(4 * columns, dtype=torch.int32).reshape(4, columns) + 1
    ).cuda()
    loops, slots, sizes, cumulative, maximums, bounds = (
        build_chunked_prefill_metadata_arrays(lengths.cuda(), lengths, table, page_size)
    )
    assert loops == 2
    assert max(s.numel() for s in slots) <= 32768
    assert maximums == (bounds[1:] - bounds[:-1]).max(dim=1).values.tolist()
    torch.testing.assert_close(sizes.cpu(), bounds[1:] - bounds[:-1])
    for request, length in enumerate(lengths.tolist()):
        got = torch.cat(
            [
                s[cumulative[i, request] : cumulative[i, request + 1]]
                for i, s in enumerate(slots)
            ]
        )
        position = torch.arange(length, device="cuda")
        expected = (
            table[request, position // page_size] * page_size + position % page_size
        )
        torch.testing.assert_close(got.long(), expected)
    starts = bounds[:-1].cuda()

    def fill():
        for i, output in enumerate(slots):
            create_chunked_cache_kv_indices_paged[(4,)](
                table,
                starts[i],
                sizes[i],
                cumulative[i],
                output,
                table.stride(0),
                page_size,
            )

    fill()
    expected = [s.clone() + page_size for s in slots]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fill()
    table.add_(1)
    graph.replay()
    for actual, reference in zip(slots, expected):
        torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_page_table_width_does_not_compile_a_new_kernel():
    helper = Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test/utils.py"
    spec = importlib.util.spec_from_file_location("kernel_test_utils", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    starts = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    sizes = torch.tensor([64, 64], device="cuda", dtype=torch.int32)
    cumulative = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    output = torch.empty(128, device="cuda", dtype=torch.int32)

    def fill(width):
        table = torch.arange(2 * width, device="cuda", dtype=torch.int32).reshape(
            2, width
        )
        create_chunked_cache_kv_indices_paged[(2,)](
            table, starts, sizes, cumulative, output, width, 64
        )
        torch.testing.assert_close(
            output,
            torch.cat(
                (
                    torch.arange(64, device="cuda"),
                    torch.arange(64, device="cuda") + width * 64,
                )
            ).int(),
        )

    fill(16)
    with module.assert_no_triton_compile(create_chunked_cache_kv_indices_paged):
        for width in (17, 31, 32, 47):
            fill(width)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 10,
    reason="TokenSpeed MLA prefill requires Blackwell or newer",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "prefixes,queries,capacity",
    [
        ([1, 0, 192], [1, 7, 17], 64),
        ([1, 0, 12288], [1, 7, 17], 4096),
        ([10000, 127, 20000], [13, 3, 17], 4096),
        ([35422, 34576, 253783, 54497], [13, 7, 3, 17], 32768),
    ],
)
def test_attention_partition_accuracy_and_graph_replay(
    prefixes, queries, capacity, dtype
):
    from tokenspeed_kernel.ops.attention import attn_merge_state
    from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla import (
        tokenspeed_mla_prefill,
    )

    torch.manual_seed(822)
    heads = 4
    query = torch.randn(
        sum(queries), heads, 192, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    keys = [
        torch.randn(p + n, heads, 192, device="cuda", dtype=torch.bfloat16).to(dtype)
        for p, n in zip(prefixes, queries)
    ]
    values = [
        torch.randn(p + n, heads, 128, device="cuda", dtype=torch.bfloat16).to(dtype)
        for p, n in zip(prefixes, queries)
    ]
    cuq = torch.tensor(
        [0] + list(itertools.accumulate(queries)), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(prefixes)
    width = capacity // len(prefixes)
    plans = [
        chunking(lengths, -(-max(prefixes) // width), len(prefixes), width),
        plan_prefix_chunks(lengths, capacity),
    ]

    def inputs(k, v, lengths):
        return (
            k,
            v,
            torch.tensor(lengths, device="cuda", dtype=torch.int32),
            torch.tensor(
                [0] + list(itertools.accumulate(lengths)),
                device="cuda",
                dtype=torch.int32,
            ),
            max(lengths),
        )

    causal = inputs(
        torch.cat([k[p:] for k, p in zip(keys, prefixes)]),
        torch.cat([v[p:] for v, p in zip(values, prefixes)]),
        queries,
    )
    prepared = []
    for plan in plans:
        prepared.append(
            [
                inputs(
                    torch.cat([k[a:b] for k, a, b in zip(keys, starts, ends)]),
                    torch.cat([v[a:b] for v, a, b in zip(values, starts, ends)]),
                    sizes,
                )
                for starts, ends, sizes in zip(
                    plan.starts.tolist(), plan.ends.tolist(), plan.len_in_chunk.tolist()
                )
            ]
        )

    def attend(data, causal):
        k, v, sizes, cu, maximum = data
        return tokenspeed_mla_prefill(
            query=query,
            key=k,
            value=v,
            seq_lens=sizes,
            cum_seq_lens=cu,
            max_seq_len=maximum,
            batch_size=len(prefixes),
            softmax_scale=192**-0.5,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cuq,
            max_seq_len_q=max(queries),
        )

    def run(plan):
        output, lse = attend(causal, True)
        for data in prepared[plan]:
            partial, partial_lse = attend(data, False)
            attn_merge_state(output, lse, partial, partial_lse, inplace=True)
        return output, lse

    before, before_lse = run(0)
    after, after_lse = run(1)
    torch.testing.assert_close(after, before, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(after_lse, before_lse, atol=2e-4, rtol=2e-4)
    if dtype == torch.bfloat16:
        offset = 0
        for prefix, count, k, v in zip(prefixes, queries, keys, values):
            scores = (
                torch.einsum(
                    "qhd,khd->hqk", query[offset : offset + count].double(), k.double()
                )
                * 192**-0.5
            )
            allowed = (
                torch.arange(k.shape[0], device="cuda")[None, :]
                <= prefix + torch.arange(count, device="cuda")[:, None]
            )
            scores.masked_fill_(~allowed[None, :, :], float("-inf"))
            reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.double())
            torch.testing.assert_close(
                after[offset : offset + count].double(), reference, atol=3e-3, rtol=1e-2
            )
            offset += count
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_lse = run(1)
    query.copy_(torch.randn_like(query, dtype=torch.bfloat16).to(dtype))
    expected, expected_lse = run(1)
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    torch.testing.assert_close(captured_lse, expected_lse, atol=0, rtol=0)
