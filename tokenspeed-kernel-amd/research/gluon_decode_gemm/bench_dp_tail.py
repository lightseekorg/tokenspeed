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

"""Prototype: remove grid quantization with "full rounds + split-K tail".

Power-of-two tiles rarely tile N into a multiple of the 256 gfx950 CUs (for
example gate_up N=34816 with BLOCK_N=128 gives 272 tiles: one full round plus a
16-tile second round). This script runs the unchanged 3-stage kernel twice per
GEMM: a main launch over the columns that form whole rounds of 256 tiles, and a
split-K launch over the leftover columns so the tail also spans many CUs. Both
launches are captured into the same HIP graph.

The main launch alone shows what the kernel reaches once every CU is busy; the
gap between "main + tail" and "main only" is the cost a fused persistent
kernel (data-parallel rounds + Stream-K tail) would remove.

Usage: python bench_dp_tail.py OUT.jsonl [--tuned-file CSV]
"""

import argparse
import json
import math

import torch
from bench_gl3 import SHAPES, bench_graph
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import mm_3stage as G3

# (shape, M) -> {name: [(n_start, n_end, cfg), ...]}, cfg = (BM, BN, BK, WM, WN, NS, SK, XCD[, WP])
CASES = {
    ("gate_up", 64): {
        "main_only(N=32768)": [(0, 32768, (64, 128, 64, 2, 2, 3, 1, 8))],
        "main+tail_bn64_sk8": [
            (0, 32768, (64, 128, 64, 2, 2, 3, 1, 8)),
            (32768, 34816, (64, 64, 64, 2, 2, 3, 8, 8)),
        ],
    },
    ("gate_up", 128): {
        "main_only(N=32768)": [(0, 32768, (128, 128, 64, 4, 2, 3, 1, 8, 1))],
    },
    ("gate_up", 256): {
        "main_only(N=32768)": [(0, 32768, (256, 128, 64, 4, 2, 3, 1, 1, 1))],
    },
    ("gate_up", 512): {
        "main+tail_sk4": [
            (0, 32768, (256, 128, 64, 4, 2, 3, 1, 8, 1)),
            (32768, 34816, (256, 128, 64, 4, 2, 3, 4, 8)),
        ],
    },
    ("gdn_in", 64): {
        "main_only(N=16384)": [(0, 16384, (64, 64, 128, 2, 2, 3, 1, 8))],
        "main+tail_sk8": [
            (0, 16384, (64, 64, 128, 2, 2, 3, 1, 8)),
            (16384, 16480, (64, 64, 128, 2, 2, 3, 8, 8)),
        ],
    },
    ("gdn_in", 128): {
        "main+tail_sk8": [
            (0, 16384, (128, 64, 128, 2, 2, 3, 1, 8)),
            (16384, 16480, (128, 64, 128, 2, 2, 3, 8, 8)),
        ],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tuned-file", default="")
    ap.add_argument("--cold-bytes", type=float, default=640e6)
    args = ap.parse_args()
    if args.tuned_file:
        tunable = torch.cuda.tunable
        tunable.tuning_enable(False)
        tunable.record_untuned_enable(False)
        tunable.set_filename(args.tuned_file, insert_device_ordinal=False)
        tunable.enable(True)
        assert tunable.read_file(args.tuned_file), f"cannot read {args.tuned_file}"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    fout = open(args.out, "a")
    for (shape, M), variants in CASES.items():
        N, K = SHAPES[shape]
        ncopy = max(2, math.ceil(args.cold_bytes / (N * K * 2)))
        Ws = [
            torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
            for _ in range(ncopy)
        ]
        Cs = [torch.empty(M, N, device=dev, dtype=torch.bfloat16) for _ in range(ncopy)]
        A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = A.double() @ Ws[0].double().t()
        scale = ref.abs().max().item()
        rec = dict(shape=shape, M=M)
        torch_fns = [
            (lambda W=W, C=C: torch.mm(A, W.t(), out=C)) for W, C in zip(Ws, Cs)
        ]
        rec["torch_mm_us"] = round(bench_graph(torch_fns), 2)
        for name, segs in variants.items():
            assert all(G3.config_fits(c[:8], M, b - a, K) for a, b, c in segs), name

            def run(W, C, segs=segs):
                for a, b, c in segs:
                    wp = bool(c[8]) if len(c) > 8 else False
                    G3.launch_gluon_mm_a16w16_3stage_gfx950(
                        A, W[a:b], C[:, a:b], c[:8], wp, schedule_hint="mfma-schedule"
                    )

            Cs[0].fill_(float("nan"))
            run(Ws[0], Cs[0])
            torch.cuda.synchronize()
            covered = slice(segs[0][0], segs[-1][1])
            err = (
                Cs[0][:, covered].double() - ref[:, covered]
            ).abs().max().item() / scale
            us = bench_graph([(lambda W=W, C=C: run(W, C)) for W, C in zip(Ws, Cs)])
            rec[name] = dict(
                us=round(us, 2),
                relerr=err,
                programs=[
                    math.ceil(M / c[0]) * math.ceil((b - a) / c[1]) * c[6]
                    for a, b, c in segs
                ],
            )
        print(json.dumps(rec), flush=True)
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        del Ws, Cs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
