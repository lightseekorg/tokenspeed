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

"""Gluon 3-stage / Stream-K GEMMs vs hipBLASLt default vs TunableOp-tuned hipBLASLt.

Qwen3.8-27B TP1 BF16 verify projections, C = A @ W.T, cold weights (rotating
copies larger than the last-level cache) under HIP-graph replay, median of trials.

Modes:
  --mode torch                    torch.mm default (hipBLASLt heuristic)
  --mode tuned --tuned-file CSV   torch.mm replaying a TunableOp results file
  --mode gl3                      Gluon 3-stage kernel, config sweep (or --cfgs JSON)
  --mode sk                       Gluon Stream-K kernel, config sweep (or --cfgs JSON)
  --compile-only                  launch every candidate once (fills the Triton cache)

Each record is one JSON line: shape, M, method, us, relerr (max error / max |ref|).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import signal
import statistics
import time

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
ROWS = (64, 128, 192, 256, 384, 512)


def gl3_candidates(M, N, K, ns_list, wp_list, xcd_list=(1, 8)):
    out = []
    bms = [b for b in (64, 128, 256) if b < 2 * M or b == 64]
    for bm, bn, bk in itertools.product(bms, (64, 128, 256), (64, 128)):
        for wm, wn in ((2, 4), (4, 2), (2, 2), (1, 4), (4, 1)):
            tiles = math.ceil(M / bm) * math.ceil(N / bn)
            # SPLIT_K=3 fits K=6144 (o): 80 tiles x 3 = 240 programs, one round on 256 CUs.
            for sk in (1, 2, 3, 4):
                if sk > 1 and tiles >= 256:
                    continue
                if tiles * sk < 64:
                    continue
                for ns in ns_list:
                    for xcd in xcd_list:
                        cfg = (bm, bn, bk, wm, wn, ns, sk, xcd)
                        if not G3.config_fits(cfg, M, N, K):
                            continue
                        # acc registers per lane (fp32); keep below ~128
                        if bm * bn // (64 * wm * wn) > 128:
                            continue
                        for wp in wp_list:
                            out.append(cfg + (wp,))
    return out


def sk_candidates(M, N, K, ns_list, grids, xcds):
    out = []
    tiles = [
        (64, 64, 128),
        (64, 128, 64),
        (64, 128, 128),
        (64, 256, 64),
        (128, 64, 128),
        (128, 128, 64),
        (128, 256, 64),
        (256, 128, 64),
        (128, 128, 128),
        (64, 64, 64),
        (64, 128, 32),
        (64, 256, 32),
        (128, 128, 32),
        (64, 64, 32),
    ]
    for bm, bn, bk in tiles:
        if bm > 64 and bm >= 2 * M:
            continue
        for wm, wn in ((2, 4), (4, 2), (2, 2), (1, 4), (4, 1)):
            if bm * bn // (64 * wm * wn) > 128:
                continue
            for ns in ns_list:
                for gr in grids:
                    for x in xcds:
                        c = (bm, bn, bk, wm, wn, ns, gr, x)
                        if SK.config_fits(c, M, N, K):
                            out.append(c)
    return out


def bench_graph(fns, reps=4, trials=9):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in fns:
                f()
    g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(trials):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / (reps * len(fns)))
    del g
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("torch", "tuned", "gl3", "sk"))
    ap.add_argument("--tuned-file", default="")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument(
        "--cfgs", default="", help="JSON {shape: {M: [cfg,...]}} or [cfg,...]"
    )
    ap.add_argument("--ns", default="3")
    ap.add_argument("--wp", default="0,1")
    ap.add_argument("--xcd", default="8")
    ap.add_argument("--grid", default="256,512")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cold-bytes", type=float, default=640e6)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--crash-file", default="")
    ap.add_argument("--pending-file", default="")
    ap.add_argument("--alarm", type=int, default=300)
    ap.add_argument(
        "--variants",
        default="base",
        help="comma list of base|sched|pin|sched+pin (MFMA scheduler / AGPR pins)",
    )
    args = ap.parse_args()
    variants = args.variants.split(",")

    def vkw(v):
        return dict(
            schedule_hint="mfma-schedule" if "sched" in v else "", pin_acc="pin" in v
        )

    if args.mode == "tuned":
        # Replay a TunableOp results file: lookups only, no tuning on a miss.
        tunable = torch.cuda.tunable
        tunable.tuning_enable(False)
        tunable.record_untuned_enable(False)
        tunable.set_filename(args.tuned_file, insert_device_ordinal=False)
        tunable.enable(True)
        assert tunable.read_file(args.tuned_file), f"cannot read {args.tuned_file}"

    part, nparts = map(int, args.part.split("/"))
    ns_list = tuple(map(int, args.ns.split(",")))
    wp_list = tuple(bool(int(x)) for x in args.wp.split(","))
    explicit = json.loads(args.cfgs) if args.cfgs else None
    crashed = set()
    # A config that killed the previous process (left in the pending file) is
    # recorded as crashed so a relaunch skips it.
    if args.pending_file and args.crash_file and os.path.exists(args.pending_file):
        pend = open(args.pending_file).read().strip()
        if pend:
            with open(args.crash_file, "a") as cf:
                cf.write(pend + "\n")
        os.remove(args.pending_file)
    if args.crash_file and os.path.exists(args.crash_file):
        crashed = {l.strip() for l in open(args.crash_file) if l.strip()}
    done = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                done.add(f"{r['shape']}|{r['M']}|{r['method']}")
            except Exception:  # noqa: BLE001
                pass

    dev = torch.device("cuda")
    torch.manual_seed(0)
    fout = open(args.out, "a")
    job = 0
    for name in args.shapes.split(","):
        N, K = SHAPES[name]
        ncopy = (
            1 if args.compile_only else max(2, math.ceil(args.cold_bytes / (N * K * 2)))
        )
        Ws = [
            torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
            for _ in range(ncopy)
        ]
        for M in map(int, args.rows.split(",")):
            A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            Cs = [
                torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                for _ in range(ncopy)
            ]
            ref = A.double() @ Ws[0].double().t()
            scale = ref.abs().max().item()
            cands = []
            if args.mode == "sk":
                if isinstance(explicit, dict):
                    cfgs = explicit.get(name, {}).get(str(M), [])
                elif explicit is not None:
                    cfgs = explicit
                else:
                    cfgs = sk_candidates(
                        M,
                        N,
                        K,
                        ns_list,
                        tuple(map(int, args.grid.split(","))),
                        tuple(map(int, args.xcd.split(","))),
                    )
                for cfg in cfgs:
                    cfg = tuple(cfg)
                    if not SK.config_fits(cfg, M, N, K):
                        continue
                    for v in variants:
                        cands.append(
                            (
                                "sk:" + ",".join(map(str, cfg)) + f"@{v}",
                                (
                                    lambda c, kw: lambda W, C: SK.launch_gluon_mm_a16w16_streamk_gfx950(
                                        A, W, C, c, **kw
                                    )
                                )(cfg, vkw(v)),
                            )
                        )
            elif args.mode in ("torch", "tuned"):
                cands.append((args.mode, lambda W, C: torch.mm(A, W.t(), out=C)))
            else:
                if isinstance(explicit, dict):
                    cfgs = explicit.get(name, {}).get(str(M), [])
                elif explicit is not None:
                    cfgs = explicit
                else:
                    cfgs = gl3_candidates(
                        M, N, K, ns_list, wp_list, tuple(map(int, args.xcd.split(",")))
                    )
                for cfg in cfgs:
                    cfg = tuple(cfg)
                    if not G3.config_fits(cfg[:8], M, N, K):
                        continue
                    wp = bool(cfg[8]) if len(cfg) > 8 else False
                    for v in variants:
                        cands.append(
                            (
                                "gl3:"
                                + ",".join(map(str, cfg[:8]))
                                + f",{int(wp)}@{v}",
                                (
                                    lambda c, wp, kw: lambda W, C: G3.launch_gluon_mm_a16w16_3stage_gfx950(
                                        A, W, C, c[:8], wp, **kw
                                    )
                                )(cfg, wp, vkw(v)),
                            )
                        )
            for label, fn in cands:
                job += 1
                if job % nparts != part:
                    continue
                ck = f"{K}|{N}|{label}"
                if ck in crashed or f"{name}|{M}|{label}" in done:
                    continue
                if args.pending_file:
                    with open(args.pending_file, "w") as pf:
                        pf.write(ck + "\n")
                rec = {"shape": name, "M": M, "N": N, "K": K, "method": label}
                signal.alarm(args.alarm)
                try:
                    t0 = time.time()
                    Cs[0].fill_(float("nan"))
                    fn(Ws[0], Cs[0])
                    torch.cuda.synchronize()
                    rec["first_s"] = round(time.time() - t0, 2)
                    err = ((Cs[0].double() - ref).abs().max().item()) / scale
                    rec["relerr"] = err
                    if not math.isfinite(err) or err > args.tol:
                        rec["bad"] = True
                    elif not args.compile_only:
                        fns = [
                            (lambda W, C: lambda: fn(W, C))(W, C)
                            for W, C in zip(Ws, Cs)
                        ]
                        rec["us"] = round(bench_graph(fns), 2)
                        # repeat-launch correctness (split-K counters must self-reset)
                        fn(Ws[0], Cs[0])
                        torch.cuda.synchronize()
                        err2 = ((Cs[0].double() - ref).abs().max().item()) / scale
                        rec["relerr2"] = err2
                        if not math.isfinite(err2) or err2 > args.tol:
                            rec["bad"] = True
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                signal.alarm(0)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                print(json.dumps(rec), flush=True)
            del A, Cs, ref
        del Ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
