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

"""Cold-L2 comparison of Blackwell QSA score/top-k implementations.

The benchmark clears the device cache before every invocation. CUDA events
bracket only score/top-k, so cache clearing is not included in reported time.
Unequal ``--query-lengths`` use ragged runs; equal lengths select uniform runs.
Use ``--max-partial-mb 0`` to benchmark the workspace-free grouped fallback.
KV token counts may leave an incomplete ratio-four compressed block.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.numerics.torch.attention.qsa import torch_qsa_paged_block_scores
from tokenspeed_kernel.ops.attention.dsa.cuda import has_ragged_decode_topk
from tokenspeed_kernel.ops.attention.qsa import qwen4_exp_qsa_block_topk
from tokenspeed_kernel.platform import pdl_enabled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument(
        "--pages",
        type=int,
        required=True,
        help="Per-request page-table capacity in 64-block compressed pages",
    )
    parser.add_argument(
        "--kv-tokens",
        type=int,
        help="Live uncompressed KV length; omission uses the page-table capacity",
    )
    layout = parser.add_mutually_exclusive_group(required=True)
    layout.add_argument("--queries-per-request", type=int)
    layout.add_argument(
        "--query-lengths",
        type=int,
        nargs="+",
        help="Consecutive request query counts; equal lengths select uniform, unequal lengths ragged",
    )
    parser.add_argument(
        "--solution", choices=("auto", "cute_dsl", "triton", "logits"), required=True
    )
    parser.add_argument("--max-logits-mb", type=int, required=True)
    parser.add_argument(
        "--max-partial-mb",
        type=int,
        required=True,
        help="Streaming scratch budget in MiB; zero disables split scratch",
    )
    parser.add_argument("--timing", choices=("eager", "graph"), required=True)
    parser.add_argument("--frontier", choices=("full", "causal"), required=True)
    parser.add_argument("--distribution", choices=("random", "ties"), required=True)
    parser.add_argument("--pdl", choices=("on", "off"), required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=83)
    return parser.parse_args()


def _kernel_resources(graph: torch.cuda.CUDAGraph) -> list[dict]:
    from cuda.bindings import driver as cu

    def checked(result):
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA resource query failed: {result}")
        return result[1:]

    raw = cu.CUgraph(graph.raw_cuda_graph())
    _, count = checked(cu.cuGraphGetNodes(raw, 0))
    nodes, _ = checked(cu.cuGraphGetNodes(raw, count))
    resources = []
    for node in nodes:
        (kind,) = checked(cu.cuGraphNodeGetType(node))
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        (params,) = checked(cu.cuGraphKernelNodeGetParams(node))
        (name,) = checked(cu.cuFuncGetName(params.func))
        if isinstance(name, bytes):
            name = name.decode()
        attributes = {}
        for label, attribute in (
            (
                "registers_per_thread",
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            ),
            (
                "static_shared_bytes",
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            ),
            (
                "local_bytes_per_thread",
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            ),
        ):
            (attributes[label],) = checked(
                cu.cuFuncGetAttribute(attribute, params.func)
            )
        threads = params.blockDimX * params.blockDimY * params.blockDimZ
        (active_blocks,) = checked(
            cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                params.func, threads, params.sharedMemBytes
            )
        )
        resources.append(
            {
                "name": name,
                "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                "threads": threads,
                "dynamic_shared_bytes": params.sharedMemBytes,
                "max_active_blocks_per_sm": active_blocks,
                **attributes,
            }
        )
    return resources


def _validate_selection(
    selected: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
) -> int:
    sample = torch.linspace(
        0, query.shape[0] - 1, min(8, query.shape[0]), device=query.device
    ).long()
    scores = torch_qsa_paged_block_scores(
        query[sample],
        key_cache,
        page_table,
        request_indices[sample].long(),
        complete_blocks[sample],
        page_size=page_size,
    )
    ids = selected[sample]
    torch.testing.assert_close(
        (ids >= 0).sum(1), complete_blocks[sample].clamp_max(block_topk).long()
    )
    assert ((ids == -1) | ((ids >= 0) & (ids < complete_blocks[sample, None]))).all()
    ordered = ids.sort(1).values
    assert ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all()
    actual = scores.gather(1, ids.clamp_min(0).long())
    actual.masked_fill_(ids < 0, -float("inf"))
    if scores.shape[1] < block_topk:
        scores = torch.nn.functional.pad(
            scores, (0, block_topk - scores.shape[1]), value=-float("inf")
        )
    torch.testing.assert_close(
        actual.sort(1, descending=True).values,
        scores.topk(block_topk, dim=1).values,
        atol=1e-4,
        rtol=1e-5,
    )
    return sample.numel()


def main() -> None:
    args = _parse_args()
    if args.rows < 1 or args.pages < 1:
        raise ValueError("rows and pages must be positive")
    if args.warmups < 1 or args.iterations < 1:
        raise ValueError("warmups and iterations must be positive")
    if args.max_partial_mb < 0:
        raise ValueError("max-partial-mb must be non-negative")
    if args.query_lengths is not None:
        lengths = args.query_lengths
        if min(lengths) < 1 or sum(lengths) != args.rows:
            raise ValueError("query-lengths must be positive and sum to rows")
        width = lengths[0] if len(set(lengths)) == 1 else None
    else:
        width = args.queries_per_request
        if width < 1 or args.rows % width:
            raise ValueError("queries-per-request must be positive and divide rows")
        lengths = [width] * (args.rows // width)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("QSA stream score/top-k benchmark requires Blackwell")

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    page_size = 64
    block_topk = 512
    head_dim = 128
    requests = len(lengths)
    kv_tokens = args.pages * page_size * 4 if args.kv_tokens is None else args.kv_tokens
    if kv_tokens < 1 or kv_tokens > args.pages * page_size * 4:
        raise ValueError("kv-tokens must be positive and fit the table capacity")
    live_blocks = kv_tokens // 4
    live_pages = math.ceil(live_blocks / page_size)
    query_storage = torch.randn(
        args.rows, 5, head_dim, dtype=torch.bfloat16, device=device
    )
    query = query_storage[:, :4]
    if args.distribution == "ties":
        query.zero_()
    key_cache = torch.randn(
        (1 + max(1024, requests * live_pages)) * page_size,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    page_table = torch.zeros((requests, args.pages), dtype=torch.int32, device=device)
    page_table[:, :live_pages] = torch.arange(
        1,
        1 + requests * live_pages,
        dtype=torch.int32,
        device=device,
    ).reshape(requests, live_pages)
    request_indices = torch.arange(
        requests, dtype=torch.int32, device=device
    ).repeat_interleave(torch.tensor(lengths, device=device))
    complete_blocks = torch.full(
        (args.rows,),
        live_blocks,
        dtype=torch.int32,
        device=device,
    )
    if args.frontier == "causal":
        # A ratio-four compressed cache: a fresh prefill completes one block
        # per four tokens. Shorter chunks start after a cached prefix.
        complete_blocks = torch.cat(
            [
                (
                    (
                        max(0, kv_tokens - length)
                        + torch.arange(length, device=device, dtype=torch.int32)
                        + 1
                    )
                    // 4
                ).clamp_max(live_blocks)
                for length in lengths
            ]
        )
    enable_pdl = args.pdl == "on"
    pdl_enabled(enable_pdl)
    if args.solution in ("auto", "logits") and not has_ragged_decode_topk():
        raise RuntimeError("logits comparison requires the persistent CUDA top-k")

    def invoke() -> torch.Tensor:
        tensors = query, key_cache, page_table, request_indices, complete_blocks
        path = args.solution if args.solution in ("auto", "logits") else "stream"
        backend = args.solution if path == "stream" else None
        return qwen4_exp_qsa_block_topk(
            *tensors,
            page_size=page_size,
            block_topk=block_topk,
            queries_per_request=width,
            max_partial_bytes=args.max_partial_mb << 20,
            max_logits_bytes=args.max_logits_mb << 20,
            path=path,
            solution=backend,
            override=None,
            enable_pdl=enable_pdl,
        )

    # Compile before warmup so JIT work is outside timing.
    selected = invoke()
    torch.cuda.synchronize()
    validated_rows = _validate_selection(
        selected,
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
    )
    cache_clear_buffer = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    for _ in range(args.warmups):
        triton.runtime.driver.active.clear_cache(cache_clear_buffer)
        invoke()
    torch.cuda.synchronize()

    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True, external=args.timing == "graph"),
            torch.cuda.Event(enable_timing=True, external=args.timing == "graph"),
        )
        for _ in range(args.iterations)
    ]
    # CUDA event initialization may be lazy; prime it outside measured samples.
    for start, end in event_pairs:
        start.record()
        end.record()
    torch.cuda.synchronize()
    resources = None
    if args.timing == "graph":
        start, end = event_pairs[0]
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            triton.runtime.driver.active.clear_cache(cache_clear_buffer)
            start.record()
            output = invoke()
            end.record()
        resources = _kernel_resources(graph)
        samples_us = []
        for _ in range(args.iterations):
            graph.replay()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000.0)
    else:
        for start, end in event_pairs:
            triton.runtime.driver.active.clear_cache(cache_clear_buffer)
            start.record()
            output = invoke()
            end.record()
        torch.cuda.synchronize()
        samples_us = [start.elapsed_time(end) * 1000.0 for start, end in event_pairs]
    ordered = sorted(samples_us)
    p90_rank = 0.9 * (len(ordered) - 1)
    p90_low = math.floor(p90_rank)
    p90_high = math.ceil(p90_rank)
    p90_us = ordered[p90_low]
    if p90_low != p90_high:
        p90_us += (ordered[p90_high] - ordered[p90_low]) * (p90_rank - p90_low)
    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "cache": "cold_l2_per_invocation",
                "path": args.solution,
                "timing": args.timing,
                "frontier": args.frontier,
                "distribution": args.distribution,
                "device": properties.name,
                "l2_cache_bytes": properties.L2_cache_size,
                "rows": args.rows,
                "blocks": args.pages * page_size,
                "kv_tokens": kv_tokens,
                "physical_key_pages": key_cache.shape[0] // page_size,
                "queries_per_request": width,
                "query_lengths": args.query_lengths,
                "max_logits_mb": args.max_logits_mb,
                "max_partial_mb": args.max_partial_mb,
                "pdl": enable_pdl,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "seed": args.seed,
                "validated_rows": validated_rows,
                "samples_us": samples_us,
                "median_us": statistics.median(samples_us),
                "mean_us": statistics.fmean(samples_us),
                "p90_us": p90_us,
                "min_us": min(samples_us),
                "max_us": max(samples_us),
                "kernel_resources": resources,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
