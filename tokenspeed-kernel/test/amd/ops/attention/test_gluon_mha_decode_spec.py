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

"""Paged grouped-query block verification and flattened non-causal drafts.

The optional script entry point measures the same complete registered operation
as the tests, including split reduction and Python launch overhead. It is not a
model benchmark. Long histories share physical cache pages across requests to
keep the isolated operator fixture small; distinct/reordered page tables still
exercise address translation.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
from tokenspeed_kernel.platform import current_platform

MAX_CONTEXT = 50176


@dataclass(frozen=True)
class DecodeCase:
    concurrency: int
    mode: str
    context: int

    @property
    def width(self):
        return 1 if self.mode in ("draft", "control") else 16

    @property
    def heads(self):
        return 5 if self.mode == "draft" else 6

    @property
    def window(self):
        return {"full": -1, "window": 511, "draft": 4095, "control": -1}[self.mode]


def make_inputs(case: DecodeCase):
    """Make strided queries like the runtime's projection view and page64 KV."""
    torch.manual_seed(417)
    width = 1 if case.mode == "control" else 16
    rows = case.concurrency * width
    pages = (case.context + 63) // 64
    table_pages = (MAX_CONTEXT + 63) // 64
    # Leave a gap after Q, as in a view into a packed projection result.
    q = torch.randn(rows, case.heads + 2, 128, dtype=torch.bfloat16, device="cuda")[
        :, : case.heads
    ]
    k = torch.randn(pages + 3, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    table = torch.stack(
        [
            (torch.arange(table_pages, device="cuda", dtype=torch.int32) + i + 1)
            % (pages + 3)
            for i in range(case.concurrency)
        ]
    )
    lengths = torch.tensor(
        [max(width + 1, case.context - (i % 4)) for i in range(case.concurrency)],
        device="cuda",
        dtype=torch.int32,
    )
    if case.mode == "draft":
        # All 16 block positions see the same block-end length: non-causal
        # attention expressed through the shared one-query metadata contract.
        table = table.repeat_interleave(16, dim=0)
        lengths = lengths.repeat_interleave(16)
    return {
        "q": q,
        "k_cache": k,
        "v_cache": v,
        "page_table": table,
        "cache_seqlens": lengths,
        "max_seqlen_k": MAX_CONTEXT,
        "max_seqlen_q": case.width,
        "window_left": case.window,
    }


@contextmanager
def reference_precision():
    """Require full FP32 matmuls without changing later tests' precision."""
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@reference_precision()
def reference(inputs):
    """Independent FP32 dense oracle; only one request is materialized at once."""
    q = inputs["q"].float()
    result = torch.empty_like(q)
    width = inputs["max_seqlen_q"]
    lengths = inputs["cache_seqlens"].cpu().tolist()
    for batch, length in enumerate(lengths):
        blocks = inputs["page_table"][batch, : (length + 63) // 64].long()
        k = inputs["k_cache"][blocks].reshape(-1, 128)[:length].float()
        v = inputs["v_cache"][blocks].reshape(-1, 128)[:length].float()
        query = q[batch * width : (batch + 1) * width]
        logits = torch.einsum("qhd,kd->qhk", query, k) * (128**-0.5)
        ends = length - width + 1 + torch.arange(width, device=q.device)
        positions = torch.arange(length, device=q.device)
        mask = positions[None, :] < ends[:, None]
        if inputs["window_left"] >= 0:
            mask &= positions[None, :] >= (ends - inputs["window_left"] - 1)[:, None]
        probabilities = torch.softmax(
            logits.masked_fill(~mask[:, None, :], -torch.inf), dim=-1
        )
        result[batch * width : (batch + 1) * width] = torch.einsum(
            "qhk,kd->qhd", probabilities, v
        )
    return result


def changed_inputs(inputs):
    """Change values only, preserving every captured address and host bound."""
    inputs["q"].mul_(-0.75)
    inputs["cache_seqlens"].sub_(1)
    table = inputs["page_table"]
    table.copy_(table.roll(1, dims=1))


def check_output(actual, expected):
    torch.testing.assert_close(actual.float(), expected, atol=0.006, rtol=0.006)


@pytest.mark.parametrize("raises", [False, True])
def test_reference_precision_is_restored(raises):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        if raises:
            with pytest.raises(ValueError, match="reference failure"):
                with reference_precision():
                    assert torch.get_float32_matmul_precision() == "highest"
                    raise ValueError("reference failure")
        else:
            with reference_precision():
                assert torch.get_float32_matmul_precision() == "highest"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("mode", ["full", "window", "draft"])
@pytest.mark.parametrize("context", [17, 63, 64, 65, 8192, 50000])
def test_spec_decode_graph_replay(concurrency, mode, context):
    if not current_platform().is_cdna4:
        pytest.skip("requires gfx950")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        gluon_mha_decode_gfx950,
        gluon_mha_decode_reduce_gfx950,
        gluon_mha_decode_sliding_gfx950,
    )
    from utils import assert_no_triton_compile

    case = DecodeCase(concurrency, mode, context)
    inputs = make_inputs(case)
    expected = reference(inputs)
    for solution in ("triton", "gluon"):
        check_output(mha_decode_with_kvcache(**inputs, solution=solution), expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mha_decode_with_kvcache(**inputs, solution="gluon")
    graph.replay()
    check_output(actual, expected)
    changed_inputs(inputs)
    expected = reference(inputs)
    with assert_no_triton_compile(
        gluon_mha_decode_gfx950,
        gluon_mha_decode_reduce_gfx950,
        gluon_mha_decode_sliding_gfx950,
    ):
        # Eager dispatch under the guard verifies dynamic values do not become
        # specialization keys; graph replay alone could not establish that.
        check_output(mha_decode_with_kvcache(**inputs, solution="gluon"), expected)
        graph.replay()
        check_output(actual, expected)
    assert torch.cuda.max_memory_allocated() < 8 * 1024**3


def paired_timing(inputs, rounds, iterations, *, graph_mode):
    """Paired alternating order, synchronized wall and GPU-event timings."""
    samples = {solution: [] for solution in ("triton", "gluon")}
    for solution in samples:
        for _ in range(3):
            mha_decode_with_kvcache(**inputs, solution=solution)
    torch.cuda.synchronize()
    graphs = {}
    outputs = {}
    if graph_mode:
        for solution in samples:
            graphs[solution] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[solution]):
                outputs[solution] = mha_decode_with_kvcache(**inputs, solution=solution)
    for round_idx in range(rounds):
        order = ("triton", "gluon") if round_idx % 2 == 0 else ("gluon", "triton")
        for solution in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            before = time.perf_counter_ns()
            start.record()
            for _ in range(iterations):
                if graph_mode:
                    graphs[solution].replay()
                else:
                    mha_decode_with_kvcache(**inputs, solution=solution)
            end.record()
            end.synchronize()
            wall_us = (time.perf_counter_ns() - before) / 1000 / iterations
            samples[solution].append(
                {
                    "wall_us": wall_us,
                    "device_us": start.elapsed_time(end) * 1000 / iterations,
                }
            )
    return {
        solution: {
            "samples": values,
            "median_wall_us": statistics.median(v["wall_us"] for v in values),
            "median_device_us": statistics.median(v["device_us"] for v in values),
        }
        for solution, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    # Only this isolated benchmark process owns a device allocation ceiling;
    # input construction must not change the allocator shared by pytest files.
    torch.cuda.set_per_process_memory_fraction(0.02)
    results = []
    cases = [
        DecodeCase(c, mode, context)
        for c in (1, 2, 4, 8, 16)
        for mode in ("full", "window", "draft")
        for context in (65, 50000)
    ]
    cases += [DecodeCase(1, "control", context) for context in (65, 50000)]
    for case in cases:
        inputs = make_inputs(case)
        expected = reference(inputs)
        for solution in ("triton", "gluon"):
            check_output(mha_decode_with_kvcache(**inputs, solution=solution), expected)
        result = {
            **case.__dict__,
            "physical_rows": inputs["q"].shape[0],
            "heads": case.heads,
            "width": case.width,
            "window_left": case.window,
            "timing": {
                execution: paired_timing(
                    inputs,
                    args.rounds,
                    args.iterations,
                    graph_mode=execution == "graph",
                )
                for execution in ("eager", "graph")
            },
        }
        results.append(result)
        args.output.write_text(
            json.dumps({"results": results, "complete": False}, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
        del inputs, expected
        gc.collect()
        torch.cuda.empty_cache()
    args.output.write_text(
        json.dumps({"results": results, "complete": True}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
