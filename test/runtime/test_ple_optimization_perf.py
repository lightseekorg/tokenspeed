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

import statistics
import time

import pytest
import torch
from tokenspeed_kernel.ops.ple import (
    ple_ngram_ids,
    ple_page_gather,
    ple_page_gather_pair,
)
from tokenspeed_kernel.ops.ple.triton import _ngram_ids_kernel
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.layers.qwen4_exp_ple import (
    _UNIFORM_INDEX_CACHE,
    Qwen4ExpPLELayer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _measure(fn, mode: str) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    if mode == "graph":
        _UNIFORM_INDEX_CACHE.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            retained = [fn() for _ in range(16)]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 1600)
        assert retained
        return statistics.median(samples)

    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 100)
    return statistics.median(samples)


def _compare(name: str, baseline, optimized, mode: str) -> None:
    baseline_us = _measure(baseline, mode)
    optimized_us = _measure(optimized, mode)
    assert baseline_us > 0 and optimized_us > 0
    print(
        f"PLE_PERF {name} {mode}: baseline={baseline_us:.3f} us "
        f"optimized={optimized_us:.3f} us speedup={baseline_us / optimized_us:.3f}x"
    )


def _ngram_inputs(batch_size: int, length: int):
    device = torch.device("cuda")
    total = batch_size * length
    input_ids = torch.arange(1, total + 1, device=device, dtype=torch.long)
    initial = torch.full((batch_size, 2), 7, device=device, dtype=torch.long)
    multipliers = torch.tensor([3, 5, 7], device=device, dtype=torch.long)
    vocab_sizes = torch.full((8,), 1013, device=device, dtype=torch.long)
    offsets = torch.arange(8, device=device, dtype=torch.long) * 1013
    return input_ids, initial, multipliers, vocab_sizes, offsets


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("batch_size,length", [(1, 1), (4, 1), (4, 4)])
def test_ple_uniform_index_perf(batch_size: int, length: int, mode: str) -> None:
    input_ids, initial, multipliers, vocab_sizes, offsets = _ngram_inputs(
        batch_size, length
    )
    lengths = [length] * batch_size
    device = input_ids.device

    def lookup(index, uniform_length):
        req, col, lengths_t, starts, _, _, _ = index
        ids, _ = ple_ngram_ids(
            input_ids,
            initial,
            req,
            col,
            lengths_t,
            starts,
            multipliers,
            vocab_sizes,
            offsets,
            ngram_size=3,
            heads_per_ngram=4,
            eos_token_id=0,
            uniform_length=uniform_length,
            mod_reciprocals=None,
        )
        return ids, req, col, lengths_t, starts

    def baseline():
        return lookup(Qwen4ExpPLELayer._batch_indices(lengths, device), 0)

    def optimized():
        index, uniform_length = Qwen4ExpPLELayer._prefetch_indices(lengths, device)
        return lookup(index, uniform_length)

    for actual, expected in zip(optimized(), baseline(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _compare(f"uniform_index_bs{batch_size}_len{length}", baseline, optimized, mode)


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_ple_page_gather_pair_perf(batch_size: int, mode: str) -> None:
    context = torch.arange(64, device="cuda", dtype=torch.long).reshape(32, 2)
    conv = torch.randn(32, 4, 2560, device="cuda", dtype=torch.bfloat16)
    pages = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int32)

    def baseline():
        return (
            ple_page_gather(context, pages, context.stride(0), 99),
            ple_page_gather(conv, pages, conv.stride(0)),
        )

    def optimized():
        return ple_page_gather_pair(
            context, conv, pages, context.stride(0), conv.stride(0), 99
        )

    for actual, expected in zip(optimized(), baseline(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _compare(f"page_gather_pair_bs{batch_size}", baseline, optimized, mode)


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("batch_size,length", [(1, 1), (4, 1), (4, 4)])
def test_ple_ngram_launch_config_perf(batch_size: int, length: int, mode: str) -> None:
    input_ids, initial, multipliers, vocab_sizes, offsets = _ngram_inputs(
        batch_size, length
    )
    req, col, lengths_t, starts, _, total, _ = Qwen4ExpPLELayer._batch_indices(
        [length] * batch_size, input_ids.device
    )
    out = torch.empty((total, 8), device="cuda", dtype=torch.long)
    use_pdl = pdl_enabled()

    def launch(block: int, warps: int):
        _ngram_ids_kernel[((total * 8 + block - 1) // block,)](
            input_ids,
            initial,
            req,
            col,
            lengths_t,
            starts,
            multipliers,
            vocab_sizes,
            offsets,
            vocab_sizes,
            out,
            out,
            total,
            batch_size,
            0,
            0,
            N=3,
            HPN=4,
            H=8,
            UNIFORM_LENGTH=0,
            WRITE_TAIL=False,
            SCATTER_TAIL=False,
            USE_RECIPROCAL=False,
            ENABLE_PDL=use_pdl,
            BLOCK=block,
            num_warps=warps,
            **({"launch_pdl": True} if use_pdl else {}),
        )
        return out

    launch(256, 4)
    expected = out.clone()
    block = min(256, max(32, 1 << (total - 1).bit_length()))
    warps = 1 if block <= 32 else 4
    launch(block, warps)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _compare(
        f"ngram_launch_bs{batch_size}_len{length}",
        lambda: launch(256, 4),
        lambda: launch(block, warps),
        mode,
    )
