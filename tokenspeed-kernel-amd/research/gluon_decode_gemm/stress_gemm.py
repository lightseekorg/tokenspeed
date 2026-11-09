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

"""Repeat-launch correctness stress for the Gluon 3-stage and Stream-K GEMMs.

Split-K and Stream-K programs reduce FP32 partials through per-tile counters
that the last arriving program resets. Running each config many times in a row
catches counter races and stale-counter bugs that a single launch misses.

Usage: python stress_gemm.py {gl3,sk} CFGS_JSON ITERS
  CFGS_JSON: {shape: {M: [cfg, ...]}} (gl3_best_cfgs.json / sk_best_cfgs.json)
Prints one JSON line per (shape, M, cfg) and "TOTAL_BAD <n>" at the end.
"""

import json
import sys

import torch
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import mm_3stage as G3
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import mm_streamk as SK

SHAPES = {
    "qkv": (14336, 5120),
    "o": (5120, 6144),
    "gdn_in": (16480, 5120),
    "gate_up": (34816, 5120),
    "down": (5120, 17408),
}


def main() -> None:
    fam, path, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cfgs = json.load(open(path))
    torch.manual_seed(0)
    bad_total = 0
    for shape, rows in cfgs.items():
        N, K = SHAPES[shape]
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        for M, cfg_list in rows.items():
            M = int(M)
            A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ref = A.double() @ W.double().t()
            scale = ref.abs().max().item()
            C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            for cfg in cfg_list:
                if fam == "gl3":
                    run = lambda: G3.launch_gluon_mm_a16w16_3stage_gfx950(  # noqa: E731
                        A, W, C, tuple(cfg[:8]), bool(cfg[8])
                    )
                else:
                    run = (
                        lambda: SK.launch_gluon_mm_a16w16_streamk_gfx950(  # noqa: E731
                            A, W, C, tuple(cfg)
                        )
                    )
                first = None
                nbad = ndiff = 0
                worst = 0.0
                for _ in range(iters):
                    C.fill_(float("nan"))
                    run()
                    torch.cuda.synchronize()
                    err = (C.double() - ref).abs().max().item() / scale
                    if not err <= 1e-2:
                        nbad += 1
                    worst = max(worst, err) if err == err else float("inf")
                    if first is None:
                        first = C.clone()
                    elif not torch.equal(C, first):
                        ndiff += 1
                bad_total += nbad
                rec = dict(
                    shape=shape,
                    M=M,
                    cfg=cfg,
                    iters=iters,
                    bad=nbad,
                    nondeterministic=ndiff,
                    worst_relerr=worst,
                )
                print(json.dumps(rec), flush=True)
    print("TOTAL_BAD", bad_total)


if __name__ == "__main__":
    main()
