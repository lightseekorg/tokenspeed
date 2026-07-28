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

"""Eval-load-shaped test of the deferred-delta AttnRes chain (fwd_v2).

Reproduces, at kernel level, the serving pattern that broke the bs64 eval:
per layer, the multicast latent tail produces ``tail_out``, which the NEXT
mix consumes as its fused delta (``prefix += tail_out`` in place) instead of
a materialized eager add; graphs captured at several decode sizes replay in
DRAIN order (16 -> 1) interleaved with eager 1-token chains and allocator
churn, with NaN-poisoned padding rows like padded decode graphs. Every
iteration checks exact values of real rows against an fp32 + NCCL reference,
so any producer-lifetime, PDL-overlap, Lamport-state or pool-reuse corruption
surfaces as a value mismatch or a sticky CUDA error at the sync.

Fidelity gap vs serving: the attention all-reduce here is NCCL, not the
trtllm Lamport/MNNVL one-shot (its IPC workspace bring-up belongs to the
runtime layer); the tail -- the delta producer under suspicion -- is real.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attn_res import attn_res_fwd_v2

H, L, EPS = 7168, 3584, 1e-6
LAYERS = 12  # one block-write window's worth of chained in-place mixes
KB_MAX = 8


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {8, 16},
    reason="launch with torchrun world size 8 or 16",
)


def _setup():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, torch.device("cuda", rank)


def _tail_op(dev):
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    if not latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    ):
        pytest.skip("platform does not support the multicast tail")
    return KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
    )


def _tail_reference(routed, shared, rms_w, up_w):
    lat = routed.float().clone()
    dist.all_reduce(lat)
    sh = shared.float().clone()
    dist.all_reduce(sh)
    var = lat.pow(2).mean(-1, keepdim=True)
    lat_n = (lat * torch.rsqrt(var + EPS)).to(torch.bfloat16) * rms_w
    return (lat_n.float() @ up_w.float().T + sh).to(torch.bfloat16)


def _mix_reference(prefix_f32, delta_bf16, blocks, rw, nw, ow):
    """fp32 reference of one fused-delta mix; returns (h, new prefix fp32)."""
    p = (prefix_f32 + delta_bf16.float()).to(torch.bfloat16).float()
    cands = torch.cat([blocks.float(), p.unsqueeze(0)], dim=0)
    rs = torch.rsqrt(cands.pow(2).mean(-1, keepdim=True) + EPS)
    wq = (nw.float() * rw.float())[None, None, :]
    logits = (cands * rs * wq).sum(-1)
    w = torch.softmax(logits, dim=0)[..., None]
    mix = (cands * w).sum(0)
    ro = torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + EPS)
    return (mix * ro * ow.float()).to(torch.bfloat16), p


def _weights(dev):
    torch.manual_seed(1234)  # identical across ranks
    up_w = (torch.randn(H, L, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    rms_w = torch.randn(L, dtype=torch.bfloat16, device=dev).contiguous()
    rw = (torch.randn(H, dtype=torch.bfloat16, device=dev) * 0.05).contiguous()
    nw = (torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5).contiguous()
    ow = (torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5).contiguous()
    return up_w, rms_w, rw, nw, ow


class _ChainedStep:
    """One captured 'decode step': LAYERS x (tail -> fused-delta v2 mix)."""

    def __init__(self, op, m, dev, rank, weights, poison_last_row):
        up_w, rms_w, rw, nw, ow = weights
        torch.manual_seed(9000 + m + rank)
        self.routed = [
            (torch.randn(m, L, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
            for _ in range(LAYERS)
        ]
        torch.manual_seed(9000 + m + rank)
        self.shared = [
            (torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
            for _ in range(LAYERS)
        ]
        torch.manual_seed(500 + m)  # rank-identical residual state
        self.prefix = torch.randn(m, H, dtype=torch.bfloat16, device=dev)
        self.blocks = torch.randn(KB_MAX, m, H, dtype=torch.bfloat16, device=dev)
        if poison_last_row and m > 1:
            # Padded decode rows carry garbage; it must never leak into real
            # rows nor fault (rows are independent in every kernel involved).
            for t in (
                self.routed + self.shared + [self.blocks[k] for k in range(KB_MAX)]
            ):
                t[-1].fill_(float("nan"))
        self.prefix0 = self.prefix.clone()
        self.weights = weights
        self.op = op
        self.m = m
        self.h = torch.empty_like(self.prefix)

    def run(self):
        up_w, rms_w, rw, nw, ow = self.weights
        for i in range(LAYERS):
            kb = 1 + (i % KB_MAX)
            tail_out = self.op(self.routed[i], self.shared[i], rms_w, up_w)
            # Deferred-delta consumption: in-place prefix accumulate fused
            # into the next mix, PDL on, exactly like the serving chain.
            attn_res_fwd_v2(
                self.prefix,
                tail_out,
                self.blocks[:kb],
                rw,
                nw,
                ow,
                EPS,
                EPS,
                enable_pdl=True,
                out=self.h,
            )
        return self.h

    def reference(self):
        """fp32 reference over real row 0 only (row-independent kernels)."""
        up_w, rms_w, rw, nw, ow = self.weights
        p = self.prefix0[:1].float()
        h = None
        for i in range(LAYERS):
            kb = 1 + (i % KB_MAX)
            tail = _tail_reference(self.routed[i][:1], self.shared[i][:1], rms_w, up_w)
            h, p = _mix_reference(p, tail, self.blocks[:kb, :1], rw, nw, ow)
        return h, p.to(torch.bfloat16)

    def check(self, tag):
        ref_h, ref_p = self.reference()
        scale = max(ref_h.float().abs().max().item(), 1.0)
        err_h = (self.h[:1].float() - ref_h.float()).abs().max().item()
        err_p = (self.prefix[:1].float() - ref_p.float()).abs().max().item()
        assert err_h < 0.05 * scale, f"{tag}: mix out err {err_h} (scale {scale})"
        assert err_p < 0.05 * scale, f"{tag}: chained prefix err {err_p}"


def test_chained_delta_mix_under_drain_load():
    """Drain-order graph replays + eager 1-token chains + allocator churn."""
    rank, dev = _setup()
    op = _tail_op(dev)
    weights = _weights(dev)

    sizes = [16, 8, 4, 2, 1]
    steps = {}
    graphs = {}
    for m in sizes:
        step = _ChainedStep(op, m, dev, rank, weights, poison_last_row=True)
        # tail warmup outside capture (compiles per-m kernels, collective)
        step.prefix.copy_(step.prefix0)
        step.run()
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        step.prefix.copy_(step.prefix0)
        with torch.cuda.graph(g):
            step.run()
        steps[m], graphs[m] = step, g

    eager = _ChainedStep(op, 1, dev, rank, weights, poison_last_row=False)
    for it in range(20):
        for m in sizes:  # decode drain: shrinking batches
            step, g = steps[m], graphs[m]
            step.prefix.copy_(step.prefix0)
            g.replay()
            # a 1-token "prefill" interleaves eagerly, plus allocator churn
            junk = [torch.randn(2048, 1024, device=dev) for _ in range(2)]
            eager.prefix.copy_(eager.prefix0)
            eager.run()
            del junk
            torch.cuda.synchronize()  # surfaces any async fault at this step
            step.check(f"replay m={m} it={it}")
            eager.check(f"eager it={it}")
    dist.barrier()
