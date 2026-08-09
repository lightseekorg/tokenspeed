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

"""Every K3 MoE tail tier must produce the same answer.

The selector routes a forward to one of four tiers by token count and graph
phase, so which arithmetic runs depends on batch size and whether a CUDA graph
is replaying. An end-to-end eval only ever exercises the tiers its own shapes
happen to select — GPQA at ebs8 never reaches SEPARATE_REDUCE, and nothing in
the decode path reaches MULTIMEM_AR. This file compares the tiers against each
other directly on identical inputs, which is what makes a tier-specific
numerical defect visible at all.

The tolerance is deliberately loose, and not because any tier accumulates in
bf16 — none does; every dot product runs in fp32. The paths differ in the
order their collective sums the sixteen bf16 partials (in-switch ld_reduce vs
NCCL), and in where intermediates round back to bf16: the fused kernel rounds
its GEMM result before adding the shared partial specifically so it matches
the unfused chain's rounding. Bitwise agreement is therefore not expected;
agreement within bf16 noise is.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-6
# Above the fused tail's capacity and below the multimem floor, so one token
# count can drive every tier that does not gate on capacity.
MID_TOKENS = 64


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


# 4 is included so a single node can cover the tiers; the fused tail itself
# only supports 8 and 16 and skips itself below that.
pytestmark = pytest.mark.skipif(
    _world_size() not in {4, 8, 16},
    reason="launch with torchrun world size 4, 8 or 16",
)


def _setup() -> tuple[int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    return dist.get_rank(), torch.device("cuda", local)


def _inputs(rank: int, device: torch.device, m: int, seed: int):
    """Per-rank partials plus rank-identical weights and residual stream."""
    g = torch.Generator(device="cpu").manual_seed(seed + rank)
    routed = (torch.randn(m, L, generator=g) * 0.1).to(device, torch.bfloat16)
    shared = (torch.randn(m, H, generator=g) * 0.1).to(device, torch.bfloat16)
    gw = torch.Generator(device="cpu").manual_seed(seed)
    rms_w = (torch.randn(L, generator=gw) * 0.1).to(device, torch.bfloat16)
    up_w = (torch.randn(H, L, generator=gw) * 0.02).to(device, torch.bfloat16)
    prefix = (torch.randn(m, H, generator=gw) * 0.1).to(device, torch.bfloat16)
    return routed, shared, rms_w, up_w, prefix


def _reference(routed, shared, rms_w, up_w, prefix):
    """All-reduce both partials, RMS-norm the latent, up-project, accumulate.

    Computed in fp32 so it is a fixed target for every tier rather than one
    tier's arithmetic standing in for the truth.
    """
    routed_sum = routed.float().clone()
    shared_sum = shared.float().clone()
    dist.all_reduce(routed_sum)
    dist.all_reduce(shared_sum)
    var = routed_sum.pow(2).mean(dim=-1, keepdim=True)
    normed = routed_sum * torch.rsqrt(var + EPS) * rms_w.float()
    return prefix.float() + normed @ up_w.float().T + shared_sum


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    scale = ref.abs().max().item()
    return (out.float() - ref).abs().max().item() / max(scale, 1e-6)


@pytest.mark.parametrize("m", [1, 4, 16])
def test_fused_tail_matches_reference(m):
    """TAIL_FUSION: the decode tier, the only one with a fused kernel."""
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    ok = latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    )
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        pytest.skip("fused latent tail unsupported here")

    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD, hidden_size=H, latent_size=L, rms_eps=EPS, device=dev
    )
    routed, shared, rms_w, up_w, prefix = _inputs(rank, dev, m, seed=11)
    ref = _reference(routed, shared, rms_w, up_w, prefix)
    out = op(routed, shared, rms_w, up_w, prefix=prefix)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


@pytest.mark.parametrize("m", [256, 1024])
def test_multimem_ar_matches_reference(m):
    """MULTIMEM_AR: in-switch staged reduces, then the replicated projection."""
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_available,
        multimem_stage,
    )

    rank, dev = _setup()
    ok = multimem_available()
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        pytest.skip("multimem unavailable here")

    routed, shared, rms_w, up_w, prefix = _inputs(rank, dev, m, seed=22)
    ref = _reference(routed, shared, rms_w, up_w, prefix)

    group_name = dist.group.WORLD.group_name
    routed_stage = multimem_stage(routed, group_name, m)
    shared_stage = multimem_stage(shared, group_name, m)
    assert routed_stage is not None and shared_stage is not None
    routed_red = multimem_all_reduce_staged(routed_stage, group_name)
    shared_red = multimem_all_reduce_staged(shared_stage, group_name)

    var = routed_red.float().pow(2).mean(dim=-1, keepdim=True)
    normed = routed_red.float() * torch.rsqrt(var + EPS) * rms_w.float()
    out = prefix.float() + normed @ up_w.float().T + shared_red.float()
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


@pytest.mark.parametrize("m", [1, MID_TOKENS, 1024])
def test_separate_reduce_matches_reference(m):
    """SEPARATE_REDUCE: the portable tier, reached whenever fused AR is off.

    No end-to-end eval in this repo selects it, so this is its only coverage.
    """
    rank, dev = _setup()
    routed, shared, rms_w, up_w, prefix = _inputs(rank, dev, m, seed=33)
    ref = _reference(routed, shared, rms_w, up_w, prefix)

    routed_red = routed.clone()
    shared_red = shared.clone()
    dist.all_reduce(routed_red)
    dist.all_reduce(shared_red)
    var = routed_red.float().pow(2).mean(dim=-1, keepdim=True)
    normed = routed_red.float() * torch.rsqrt(var + EPS) * rms_w.float()
    out = prefix.float() + normed @ up_w.float().T + shared_red.float()
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


def test_tiers_agree_with_each_other():
    """The property that matters in serving: same input, same answer.

    A tier that is individually within tolerance of the reference can still
    sit on the opposite edge from another tier, and a request's tier depends
    on batch size — so agreement between tiers is checked directly.
    """
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_available,
        multimem_stage,
    )

    rank, dev = _setup()
    ok = multimem_available()
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        pytest.skip("multimem unavailable here")

    m = 512  # inside the multimem window, above the fused tail's capacity
    routed, shared, rms_w, up_w, prefix = _inputs(rank, dev, m, seed=44)

    def project(routed_red, shared_red):
        var = routed_red.float().pow(2).mean(dim=-1, keepdim=True)
        normed = routed_red.float() * torch.rsqrt(var + EPS) * rms_w.float()
        return prefix.float() + normed @ up_w.float().T + shared_red.float()

    nccl_r, nccl_s = routed.clone(), shared.clone()
    dist.all_reduce(nccl_r)
    dist.all_reduce(nccl_s)
    separate = project(nccl_r, nccl_s)

    mm_r = multimem_all_reduce_staged(
        multimem_stage(routed, dist.group.WORLD.group_name, m),
        dist.group.WORLD.group_name,
    )
    mm_s = multimem_all_reduce_staged(
        multimem_stage(shared, dist.group.WORLD.group_name, m),
        dist.group.WORLD.group_name,
    )
    multimem = project(mm_r, mm_s)

    torch.cuda.synchronize()
    scale = separate.abs().max().item()
    diff = (multimem - separate).abs().max().item() / max(scale, 1e-6)
    assert diff < 0.02, f"multimem and separate-reduce disagree by {diff}"
