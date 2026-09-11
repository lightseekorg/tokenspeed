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

"""CUDA graph benchmarks for GDN decode/norm and QSA selection/attention chains.

Run with PYTHONPATH=python:tokenspeed-kernel/python from the repository root.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from tokenspeed_kernel.ops.attention import (
    gdn_decode_mtp,
    gdn_decode_step,
    qsa_sparse_attention,
)
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_block_topk,
    qwen4_exp_qsa_selected_slots,
)
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.layers.attention.linear.layernorm_gated import rmsnorm_fn


def _gdn(batch, steps, solution):
    heads, value_heads, dim = 4, 12, 128
    q, k = [
        torch.randn(batch, steps, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    v, z = [
        torch.randn(batch, steps, value_heads, dim, device="cuda", dtype=q.dtype)
        for _ in range(2)
    ]
    a, b = [
        torch.randn(batch, steps, value_heads, device="cuda", dtype=q.dtype)
        for _ in range(2)
    ]
    weights = torch.randn(dim, device="cuda", dtype=q.dtype)
    state = torch.randn(batch * (steps + 1), value_heads, dim, dim, device="cuda")
    reads = torch.arange(batch, device="cuda", dtype=torch.int32)
    writes = torch.arange(
        batch, batch * (steps + 1), device="cuda", dtype=torch.int32
    ).reshape(batch, steps)
    args = dict(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=torch.randn(value_heads, device="cuda"),
        dt_bias=torch.randn(value_heads, device="cuda"),
        initial_state=state,
        initial_state_indices=reads,
        scale=dim**-0.5,
        use_qk_l2norm=True,
        solution=solution,
        override=None,
    )

    def forward():
        if steps == 1:
            out = gdn_decode_step(**args, output_state_indices=writes[:, 0])
        else:
            out = gdn_decode_mtp(
                **args,
                disable_state_update=False,
                output_state_indices=writes,
                intermediate_states_buffer=None,
            )
        return rmsnorm_fn(
            out.reshape(-1, dim),
            weights,
            z=z.reshape(-1, dim),
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            sigmoid_gate=False,
            weights_independent=True,
        )

    return forward


def _qsa(rows, solution):
    blocks, dim, heads, ratio, page_size = 4096, 256, 6, 4, 64
    query = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    index_query = torch.randn(rows, 4, 128, device="cuda", dtype=query.dtype)
    compressed = torch.randn(
        blocks + page_size, 1, 128, device="cuda", dtype=query.dtype
    )
    cache = [
        torch.randn(
            blocks * ratio + page_size, 1, dim, device="cuda", dtype=query.dtype
        )
        for _ in range(2)
    ]
    table = torch.arange(
        1, blocks // page_size + 1, device="cuda", dtype=torch.int32
    ).repeat(rows, 1)
    full_table = torch.arange(
        1, blocks * ratio // page_size + 1, device="cuda", dtype=torch.int32
    ).repeat(rows, 1)
    requests = torch.arange(rows, device="cuda", dtype=torch.int64)
    complete = torch.full((rows,), blocks, device="cuda", dtype=torch.int32)
    logical = complete.long() * ratio - 1

    def forward():
        selected = qwen4_exp_qsa_block_topk(
            index_query,
            compressed,
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=512,
            page_expansion=1,
            max_partial_bytes=32 * 1024 * 1024,
            solution=solution,
            persistent_topk_workspace=None,
            enable_pdl=pdl_enabled(),
        )
        slots = qwen4_exp_qsa_selected_slots(
            selected, complete, logical, requests, full_table, page_size, ratio, 2048
        )
        return qsa_sparse_attention(
            query,
            *cache,
            slots,
            scale=dim**-0.5,
            max_seqlen_q=1,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="cute_dsl",
        )

    return forward


def _measure(forward, calls, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            forward()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (calls * repeats))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("gdn", "qsa"), required=True)
    parser.add_argument(
        "--solution",
        required=True,
        help="GDN: flashinfer/triton; QSA scoring: stream/logits",
    )
    parser.add_argument("--batches", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--pdl", choices=("on", "off"), required=True)
    args = parser.parse_args()
    pdl_enabled(args.pdl == "on")
    torch.manual_seed(331)
    for batch in map(int, args.batches.split(",")):
        forward = (
            _gdn(batch, args.steps, args.solution)
            if args.operation == "gdn"
            else _qsa(batch * args.steps, args.solution)
        )
        print(
            json.dumps(
                dict(
                    operation=args.operation,
                    solution=args.solution,
                    batch=batch,
                    steps=args.steps,
                    pdl=pdl_enabled(),
                    graph_us=round(_measure(forward, 32, 100), 3),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
