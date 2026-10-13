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
"""TokenSpeed iHC kernels vs the vLLM 0.29.0 HY V4 iHC path, on the same GPU.

vLLM 0.29.0 (``vllm/models/hy_v4/nvidia/hc.py``) only uses its fused HPC iHC
kernels on sm100/sm103. On Hopper (H20/H100) it runs the eager torch bodies of
``HYV4HCPreLayer`` / ``HYV4HCPostLayer`` / ``HYV4HCHeadLayer`` under CUDA graphs
(the model is not torch.compiled), with
``torch.set_float32_matmul_precision("highest")`` (VLLM_FLOAT32_MATMUL_PRECISION
default); later releases add a Triton path for the pre and post blocks. Those
bodies are reproduced verbatim below, followed by the decoder
layer's separate ``RMSNorm`` (vLLM does not fold it into the pre block), so the
comparison is like for like with TokenSpeed's fused pre+norm.

All timings are CUDA-graph replays (launch overhead excluded, as in serving).
Run from the repository root:

    python -m test.hy_v4_parity.vllm_ihc_compare
"""

from __future__ import annotations

import argparse
import json
import statistics
from test.hy_v4_parity import gpu_kernel_check as gkc


# --------------------------------------------------------------------------- #
# vLLM eager bodies (hc.py), verbatim modulo module plumbing.
# --------------------------------------------------------------------------- #
def vllm_pre(x, hc_fn_w, hc_scale, hc_base, hc, magnitude, hc_eps, eps):
    import torch
    import torch.nn.functional as F

    shape = x.size()
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(x_flat, hc_fn_w) * rsqrt
    pre_raw = mixes[..., :hc]
    post_raw = mixes[..., hc : 2 * hc]
    pre = torch.sigmoid(pre_raw * hc_scale[0].float() + hc_base[:hc].float()) + hc_eps
    post = (
        magnitude
        * torch.sigmoid(post_raw * hc_scale[1].float() + hc_base[hc : 2 * hc].float())
        + hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)
    return y.to(x.dtype), post


def vllm_rmsnorm(x, w, eps):
    """vLLM RMSNorm.forward_cuda -> _C.rms_norm when available, else native."""
    import torch

    try:
        from vllm import _custom_ops as ops  # type: ignore

        out = torch.empty_like(x)
        ops.rms_norm(out, x, w, eps)
        return out
    except Exception:
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


def vllm_post(x, residual, post):
    dtype = x.dtype
    x = x.float()
    residual = residual.float()
    post = post.float()
    return (post.unsqueeze(-1) * x.unsqueeze(-2) + residual).to(dtype)


def vllm_head(x, w, scale, base, eps, hc_eps):
    import torch
    import torch.nn.functional as F

    shape, x_dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(x, w) * rsqrt
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)
    return y.to(x_dtype)


# --------------------------------------------------------------------------- #
def graph_time(fn, iters: int = 200, reps: int = 7) -> float:
    """Median per-call latency (us) of ``fn`` replayed from a CUDA graph that
    holds ``iters`` back-to-back calls."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / iters)
    del g
    return statistics.median(out)


def rel(got, ref) -> float:
    g, r = got.double(), ref.double()
    return ((g - r).norm() / r.norm().clamp_min(1e-30)).item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,8,16,32,64,128,256,512,1024,2048,4096,8192")
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--json", default="")
    p.add_argument("--module", default="tokenspeed_kernel.ops.residual.triton")
    args = p.parse_args()

    gkc._install_shadow_package()
    import importlib

    import torch

    torch.set_float32_matmul_precision("highest")  # vLLM default
    ts = importlib.import_module(args.module)

    dev = "cuda"
    H, hc = args.hidden, args.hc
    K = H * hc
    mag, hc_eps, eps = 2.0, 1e-6, 1e-6
    g = torch.Generator(device=dev).manual_seed(0)
    fn_pre = (torch.randn(2 * hc, K, device=dev, generator=g) * 6e-3).contiguous()
    fn_head = (torch.randn(hc, K, device=dev, generator=g) * 6e-3).contiguous()
    sc_pre = torch.tensor([0.5, 0.7], device=dev)
    bs_pre = torch.randn(2 * hc, device=dev, generator=g) * 0.3
    sc_head = torch.tensor([0.6], device=dev)
    bs_head = torch.randn(hc, device=dev, generator=g) * 0.3
    nw = (1 + 0.1 * torch.randn(H, device=dev, generator=g)).to(torch.bfloat16)

    print(f"device={torch.cuda.get_device_name()} H={H} hc={hc}")
    hdr = (
        f"{'T':>5} | {'op':<9} | {'vllm us':>9} | {'ts us':>9} | {'ts/vllm':>7} |"
        f" {'rel_vllm':>9} | {'rel_ts':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for T in [int(t) for t in args.tokens.split(",")]:
        x = (torch.randn(T, hc, H, device=dev, generator=g) * 2).to(torch.bfloat16)
        blk = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
        gates = torch.rand(T, hc, device=dev, generator=g) * 2

        # ---- reference (fp64)
        flat = x.flatten(1).double()
        rs = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        mixes = (flat @ fn_pre.double().T) * rs
        pre_r = torch.sigmoid(mixes[:, :hc] * 0.5 + bs_pre[:hc].double()) + hc_eps
        post_r = (
            mag * torch.sigmoid(mixes[:, hc:] * 0.7 + bs_pre[hc:].double()) + hc_eps
        )
        y_r = (pre_r.unsqueeze(-1) * x.double()).sum(1)
        y_r = y_r * torch.rsqrt(y_r.square().mean(-1, keepdim=True) + eps) * nw.double()
        mh = (flat @ fn_head.double().T) * rs
        ph = torch.sigmoid(mh * 0.6 + bs_head.double()) + hc_eps
        head_r = (ph.unsqueeze(-1) * x.double()).sum(1)
        post_ref = gates.double().unsqueeze(-1) * blk.double().unsqueeze(1) + x.double()

        cases = {
            "pre+norm": (
                lambda: vllm_rmsnorm(
                    vllm_pre(x, fn_pre, sc_pre, bs_pre, hc, mag, hc_eps, eps)[0],
                    nw,
                    eps,
                ),
                lambda: ts.triton_ihc_pre(
                    x, fn_pre, sc_pre, bs_pre, mag, eps, hc_eps, nw, eps
                )[0],
                y_r,
            ),
            "post": (
                lambda: vllm_post(blk, x, gates),
                lambda: ts.triton_ihc_post(blk, x, gates),
                post_ref,
            ),
            "head": (
                lambda: vllm_head(x, fn_head, sc_head, bs_head, eps, hc_eps),
                lambda: ts.triton_ihc_head(x, fn_head, sc_head, bs_head, eps, hc_eps),
                head_r,
            ),
        }
        for name, (fv, ft, ref) in cases.items():
            ev, et = rel(fv(), ref), rel(ft(), ref)
            tv, tt = graph_time(fv), graph_time(ft)
            rows.append(
                dict(T=T, op=name, vllm_us=tv, ts_us=tt, rel_vllm=ev, rel_ts=et)
            )
            print(
                f"{T:>5} | {name:<9} | {tv:>9.1f} | {tt:>9.1f} | {tt / tv:>7.2f} |"
                f" {ev:>9.2e} | {et:>9.2e}"
            )
        # post gates accuracy too
        _, post_ts = ts.triton_ihc_pre(
            x, fn_pre, sc_pre, bs_pre, mag, eps, hc_eps, nw, eps
        )
        _, post_v = vllm_pre(x, fn_pre, sc_pre, bs_pre, hc, mag, hc_eps, eps)
        print(
            f"{T:>5} | post_gate | max|ts-ref|={ (post_ts.double()-post_r).abs().max().item():.2e}"
            f"  max|vllm-ref|={(post_v.double()-post_r).abs().max().item():.2e}"
        )
        # per decoder layer: 2x(pre+norm) + 2x post
        lv = 2 * rows[-3]["vllm_us"] + 2 * rows[-2]["vllm_us"]
        lt = 2 * rows[-3]["ts_us"] + 2 * rows[-2]["ts_us"]
        print(
            f"{T:>5} | LAYER     | {lv:>9.1f} | {lt:>9.1f} | {lt / lv:>7.2f} |"
            f"  x78 layers: vllm {lv * 78 / 1000:.2f} ms, ts {lt * 78 / 1000:.2f} ms"
        )
        del x, blk, flat
        torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
