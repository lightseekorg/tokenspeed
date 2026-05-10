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

"""Micro-benchmark for the MI355 Gluon MoE GEMM.

Compares the in-tree Gluon kernel against the upstream ``triton_kernels``
matmul over a small grid of (block_m, block_n, block_k, num_warps).

Run with:
    HIP_VISIBLE_DEVICES=2 python3 -u benchmarks/moe_gluon_microbench.py

The script doesn't depend on pytest; it just prints a TSV-like table.
"""

from __future__ import annotations

import argparse
import itertools
import time

# IMPORTANT: tokenspeed_kernel must be imported before torch on the docker
# image to avoid an ABI segfault between system torch and tokenspeed_triton.
import tokenspeed_kernel  # noqa: F401  (must be first)
import torch


def _build_inputs(M, N, K, E, block_m, device="cuda"):
    from triton_kernels.tensor import make_ragged_tensor_metadata

    per_expert = max(block_m, (M // E) // block_m * block_m)
    M_padded = per_expert * E
    counts = torch.full((E,), per_expert, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M_padded)
    x = torch.randn(M_padded, K, device=device, dtype=torch.bfloat16) * 0.05
    w = torch.randn(E, K, N, device=device, dtype=torch.bfloat16) * 0.05
    return x, w, md, M_padded


def _bench(fn, args, *, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--N", type=int, default=2880)
    p.add_argument("--K", type=int, default=2880)
    p.add_argument("--E", type=int, default=128)
    p.add_argument("--top-k", type=int, default=4, dest="topk")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    from tokenspeed_kernel.ops.moe.gluon import _gluon_bf16_ragged_matmul
    from tokenspeed_kernel.ops.moe.triton_kernels import _matmul as _tk_matmul

    print(
        f"Microbench shapes: M={args.M} N={args.N} K={args.K} E={args.E} top_k={args.topk}"
    )
    print("config\ttime_ms\tTFLOPs\trel_to_triton_kernels")

    block_m_list = [32, 64, 128]
    block_n_list = [64, 128, 256]
    block_k_list = [64, 128, 256]
    num_warps_list = [4, 8]

    # First measure the upstream baseline as a reference.
    x, w, md, M_padded = _build_inputs(args.M, args.N, args.K, args.E, 64)
    flops_per_call = 2 * M_padded * args.N * args.K

    def _baseline(x_, w_, md_):
        return _tk_matmul(
            x_,
            w_,
            None,
            a_ragged_metadata=md_,
            gather_indx=None,
            scatter_indx=None,
            precision_config=None,
            fused_activation=None,
        )

    try:
        baseline_t = _bench(_baseline, (x, w, md), warmup=5, iters=20)
        print(
            f"triton_kernels (baseline)\t{baseline_t * 1000:.3f}\t"
            f"{flops_per_call / baseline_t / 1e12:.2f}\t1.00"
        )
    except Exception as e:
        print(f"triton_kernels baseline failed: {e}")
        baseline_t = float("nan")

    for bm, bn, bk, nw in itertools.product(
        block_m_list, block_n_list, block_k_list, num_warps_list
    ):
        # Skip configs that violate hardware constraints we already know.
        if (bn % 8) or (bk % 32) or (nw not in (4, 8)):
            continue
        if bn < bk and nw == 8:
            continue
        x, w, md, M_padded = _build_inputs(args.M, args.N, args.K, args.E, bm)

        def _candidate(x_, w_, md_, bm_=bm, bn_=bn, bk_=bk, nw_=nw):
            return _gluon_bf16_ragged_matmul(
                x_,
                w_,
                bias=None,
                a_ragged_metadata=md_,
                gather_indx=None,
                scatter_indx=None,
                precision_config=None,
                fused_activation=None,
                n_tokens=None,
                n_expts_act=None,
                block_m=bm_,
                block_n=bn_,
                block_k=bk_,
                num_warps=nw_,
            )

        try:
            t = _bench(_candidate, (x, w, md), warmup=5, iters=20)
            tflops = flops_per_call / t / 1e12
            rel = baseline_t / t if baseline_t == baseline_t else float("nan")
            print(
                f"bm={bm} bn={bn} bk={bk} nw={nw}\t{t * 1000:.3f}\t"
                f"{tflops:.2f}\t{rel:.2f}"
            )
        except Exception as e:
            print(
                f"bm={bm} bn={bn} bk={bk} nw={nw}\tFAILED ({type(e).__name__}: "
                f"{str(e)[:80]})"
            )


if __name__ == "__main__":
    main()
