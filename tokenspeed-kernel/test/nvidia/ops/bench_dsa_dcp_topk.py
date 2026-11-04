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

"""Cold-L2 comparison of DCP local index-candidate selectors and mask launches.

Works on DeepGEMM-shaped logits: rows padded to a multiple of 256 and passed
as the ``[:, :width]`` slice. Every row owns ``--length`` scores. The default
width is the decode page table of a 1M-token context. Every measurement is a
CUDA graph replay; the device cache is cleared before every replay and
excluded from timing.

- ``mask_<launch>_median_us`` times the length-aware mask alone: ``full_grid``
  is the launch before length-aware masking (rows x cdiv(width, 256)
  programs, most exiting past the row's length), ``prefix_<block>`` the
  device-sized grid striding ``block`` scores per iteration, and ``default``
  what ``mask_index_scores`` launches. All launches must write the same
  scores.
- ``<selector>_median_us`` times mask, selection and candidate gathering
  together. The selector is checked to have launched (a silent fallback to
  ``torch.topk`` is an error), and all selectors must return the same scores.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.dsa import deep_gemm
from tokenspeed_kernel.ops.attention.dsa._triton import index_candidates
from tokenspeed_kernel.platform import current_platform

PAGE_SIZE = 64
INITIAL_TOKENS = 4
LOCAL_TOKENS = 8
# (block, num_warps) of the length-aware launches compared with the default.
PREFIX_LAUNCHES = ((256, 4), (1024, 4), (2048, 8))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[15, 512])
    parser.add_argument("--width", type=int, default=16386 * PAGE_SIZE)
    parser.add_argument("--length", type=int, nargs="+", default=[1024, 16384, 131072])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


@contextlib.contextmanager
def _selector(name: str):
    """Force selector ``name``; yield the ragged top-k launches it makes."""
    saved = (
        deep_gemm._dcp_deep_select,
        deep_gemm._dcp_ragged_rows,
        deep_gemm.ragged_decode_topk,
    )
    launches = []
    if name != "deep_select":
        deep_gemm._dcp_deep_select = None
    if name == "torch_topk":
        deep_gemm._dcp_ragged_rows = lambda *_: None
    ragged = deep_gemm.ragged_decode_topk

    def recorded(*args, **kwargs):
        launches.append(args[0])
        return ragged(*args, **kwargs)

    deep_gemm.ragged_decode_topk = recorded
    try:
        yield launches
    finally:
        (
            deep_gemm._dcp_deep_select,
            deep_gemm._dcp_ragged_rows,
            deep_gemm.ragged_decode_topk,
        ) = saved


def _replay_us(run, reset, iters: int, cache_clear_buffer) -> tuple[float, object]:
    """Median replay time of ``run`` in a graph, and the captured result.

    ``reset`` restores the inputs before the eager warmup and every replay,
    outside the timed region.
    """
    reset()
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    samples_us = []
    for _ in range(iters):
        reset()
        triton.runtime.driver.active.clear_cache(cache_clear_buffer)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    # Keep the graph's pool alive until the caller has read the result.
    result_holder = (graph, result)
    return statistics.median(samples_us), result_holder


def _run(rows: int, width: int, length: int, topk: int, iters: int) -> dict:
    stride = triton.cdiv(width, 256) * 256
    generator = torch.Generator(device="cuda").manual_seed(rows + length)
    raw = torch.randn((rows, width), generator=generator, device="cuda")
    scores = torch.empty((rows, stride), device="cuda")[:, :width]
    pages = triton.cdiv(width, PAGE_SIZE)
    positions = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(rows, 1)
    lengths = torch.full(
        (rows, 1), min(length, width), dtype=torch.int32, device="cuda"
    )
    causal = lengths.flatten().clone()
    cache_clear_buffer = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    result = {"rows": rows, "width": width, "length": length, "topk": topk}
    mask_args = (positions, lengths, causal, PAGE_SIZE, INITIAL_TOKENS, LOCAL_TOKENS)

    def reset():
        scores.copy_(raw)

    # The mask alone, under each launch shape.
    launches = {
        "default": lambda: index_candidates.mask_index_scores(
            scores, *mask_args, mask_tail=False
        ),
        "full_grid": lambda: index_candidates._launch_mask_scores(
            scores,
            *mask_args,
            mask_tail=False,
            programs=triton.cdiv(width, 256),
            block=256,
        ),
    }
    target = triton.cdiv(8 * current_platform().sm_count, rows)
    for block, num_warps in PREFIX_LAUNCHES:
        programs = min(max(target, 1), triton.cdiv(width, block))
        launches[f"prefix_{block}_w{num_warps}"] = (
            lambda programs=programs, block=block, num_warps=num_warps: (
                index_candidates._launch_mask_scores(
                    scores,
                    *mask_args,
                    mask_tail=False,
                    programs=programs,
                    block=block,
                    num_warps=num_warps,
                )
            )
        )
    masked = None
    for name, launch in launches.items():
        median, held = _replay_us(launch, reset, iters, cache_clear_buffer)
        if masked is None:
            masked = scores.clone()
        elif not torch.equal(scores, masked):
            raise AssertionError(f"mask launch {name} wrote different scores")
        result[f"mask_{name}_median_us"] = median
        del held
    del masked

    def select():
        return deep_gemm._select_dcp_index_candidates(
            scores,
            positions,
            lengths,
            causal,
            page_size=PAGE_SIZE,
            topk=topk,
            initial_tokens=INITIAL_TOKENS,
            local_tokens=LOCAL_TOKENS,
        )

    names = ["torch_topk", "ragged"]
    if deep_gemm._dcp_deep_select is not None:
        names.append("deep_select")
    reference = None
    for name in names:
        with _selector(name) as ragged_launches:
            reset()
            select()
            # A layout the selector refuses would silently time torch.topk.
            if name == "ragged" and len(ragged_launches) != 1:
                raise AssertionError("the ragged selector did not launch")
            if name != "ragged" and ragged_launches:
                raise AssertionError(f"{name} launched the ragged selector")
            if name == "deep_select" and not deep_gemm._dcp_topk_supports_lengths(
                scores, topk
            ):
                raise AssertionError("DeepSelect refused the logits layout")
            median, held = _replay_us(select, reset, iters, cache_clear_buffer)
        _, (_, values) = held
        selected = values.sort(dim=1).values
        if reference is None:
            reference = selected
        elif not torch.equal(selected, reference):
            raise AssertionError(f"{name} selected different scores")
        result[f"{name}_median_us"] = median
        del held
    return result


def main() -> None:
    args = _parse_args()
    for rows in args.rows:
        for length in args.length:
            print(json.dumps(_run(rows, args.width, length, args.topk, args.iters)))


if __name__ == "__main__":
    main()
