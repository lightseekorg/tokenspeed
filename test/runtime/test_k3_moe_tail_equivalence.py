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

The selector routes a forward to one of the tail tiers by token count and
graph phase, so which arithmetic runs depends on batch size and whether a CUDA
graph is replaying. An end-to-end eval only ever exercises the tiers its own shapes
happen to select — GPQA at ebs8 never reaches SEPARATE_REDUCE, and nothing in
the decode path reaches MULTIMEM_AR. This file compares the tiers against each
other directly on identical inputs, which is what makes a tier-specific
numerical defect visible at all.

The tiers are driven through ``K3MoeTailComm``'s own ``_tail_*`` methods, on an
instance built by ``__new__`` with only the attributes those methods read
(the collective negotiation in ``K3MoeTailCommState`` is deliberately
bypassed). That matters: a test that re-derived norm+projection in torch
would agree with itself no matter what ``routed_norm``,
``kimi3_latent_projection_add3``, ``kimi3_join_reduce_moe`` or the fused
kernel's epilogue actually computed. Only ``mapping``, ``execution_plan`` and
``state`` are stand-ins — they carry group and policy configuration, no
arithmetic.

The tolerance is deliberately loose, and not because any tier accumulates in
bf16 — none does; every dot product runs in fp32. The paths differ in the
order their collective sums the sixteen bf16 partials (in-switch ld_reduce vs
NCCL), and in where intermediates round back to bf16: the fused kernel rounds
its GEMM result before adding the shared partial specifically so it matches
the unfused chain's rounding. Bitwise agreement is therefore not expected;
agreement within bf16 noise is.

``test_selector_boundaries`` is pure Python and runs in ordinary one-GPU CI.
Everything else needs the collective; exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-6
# Above the fused tail's capacity and below the multimem floor, so one token
# count can drive every tier that does not gate on capacity.
MID_TOKENS = 64


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _setup() -> tuple[int, torch.device]:
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    # The runtime's collectives look groups up by rank tuple; serving registers
    # them during distributed init, which this test does not run. Register the
    # already-built world group rather than calling init_process_group: that
    # would also build gloo groups, which this test never uses and which stall
    # on a multi-node launch.
    group = tuple(range(_world_size()))
    if not process_group_manager.has_process_group("nccl", group):
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    return dist.get_rank(), torch.device("cuda", local)


def _agreed(ok: bool) -> bool:
    """True only if every rank reports True — tiers must be entered together."""
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _build_comm(device: torch.device, *, latent_tail=None):
    """A ``K3MoeTailComm`` carrying just what the tail methods touch.

    Constructing it for real runs the process-wide MIN-vote and the
    collective allocators this test has no use for, so the instance is built
    by ``__new__`` and populated directly. The norm and projection are the
    production modules with random weights; ``mapping``, ``execution_plan``
    and ``state`` are namespaces because the tail reads only group handles
    and policy flags off them.
    """
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.moe.latent import Kimi3LatentProjection
    from tokenspeed.runtime.models.kimi_k3_comm import K3MoeTailComm

    world = _world_size()
    comm = object.__new__(K3MoeTailComm)
    comm.hidden_size = H
    comm.routed_hidden = L

    with torch.device(device):
        up_proj = Kimi3LatentProjection(L, H, params_dtype=torch.bfloat16)
        # RMSNorm takes its dtype from the default; the checkpoint's is bf16,
        # and its kernel rejects a weight that does not match the activation.
        norm = RMSNorm(L, eps=EPS).to(torch.bfloat16)
    # Weights are rank-identical: these are replicated in the model, and a
    # per-rank difference here would show up as a tier disagreement.
    gw = torch.Generator(device="cpu").manual_seed(7)
    up_proj.weight.data.copy_((torch.randn(H, L, generator=gw) * 0.02).to(device))
    norm.weight.data.copy_((torch.randn(L, generator=gw) * 0.1).to(device))
    comm.up_proj = up_proj
    comm.routed_norm = norm
    # Replicated projection (no shard group): the tiers take their
    # _replicated variants, matching the full-width reference below.
    comm._shard_up_projection = up_proj.shard_group is not None

    comm.mapping = SimpleNamespace(
        moe=SimpleNamespace(
            # tokenspeed groups are tuples of global ranks, not ProcessGroups.
            tp_ep_group=tuple(range(world)),
            tp_ep_size=world,
            has_tp_ep=world > 1,
        )
    )
    comm.execution_plan = SimpleNamespace(
        fused_moe_ar=True,
        lane_latent_norm_ar=False,
        comm_fusion_max_num_tokens=8192,
    )
    # Negotiated state stand-in.
    comm.state = SimpleNamespace(
        rms_eps=EPS,
        multimem_ar_ok=True,
        latent_tail_ok=latent_tail is not None,
    )
    comm.latent_tail = latent_tail
    return comm


def _inputs(rank: int, device: torch.device, m: int, seed: int):
    """Per-rank partials plus the rank-identical residual stream."""
    g = torch.Generator(device="cpu").manual_seed(seed + rank)
    routed = (torch.randn(m, L, generator=g) * 0.1).to(device, torch.bfloat16)
    shared = (torch.randn(m, H, generator=g) * 0.1).to(device, torch.bfloat16)
    gw = torch.Generator(device="cpu").manual_seed(seed)
    prefix = (torch.randn(m, H, generator=gw) * 0.1).to(device, torch.bfloat16)
    return routed, shared, prefix


def _reference(comm, routed, shared, prefix):
    """All-reduce both partials, RMS-norm the latent, up-project, accumulate.

    Computed in fp32 from the same weights the tiers use, so it is a fixed
    target rather than one tier's arithmetic standing in for the truth.
    """
    routed_sum = routed.float().clone()
    shared_sum = shared.float().clone()
    dist.all_reduce(routed_sum)
    dist.all_reduce(shared_sum)
    var = routed_sum.pow(2).mean(dim=-1, keepdim=True)
    normed = routed_sum * torch.rsqrt(var + EPS) * comm.routed_norm.weight.float()
    up_w = comm.up_proj.weight.float()
    return prefix.float() + normed @ up_w.T + shared_sum


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    scale = ref.abs().max().item()
    return (out.float() - ref).abs().max().item() / max(scale, 1e-6)


def _run_separate_reduce(comm, routed, shared, prefix, m):
    # The fork scope reduces and projects the routed side before the tier
    # method sees it; serving does that on a side stream, order unchanged.
    routed_proj = comm.reduce_project_routed(routed)
    return comm._tail_separate_reduce(routed_proj, shared, prefix, m, H)


def test_selector_boundaries():
    """The tier map itself, with no GPU or collective involved."""
    from tokenspeed.runtime.models.kimi_k3_comm import (
        K3MoETailTier,
        select_k3_moe_tail_tier,
    )

    def pick(num_tokens, *, graph_phase=True, fused_max=16, fused_ar=True, mm=True):
        return select_k3_moe_tail_tier(
            num_tokens=num_tokens,
            graph_phase=graph_phase,
            tail_fusion_max_tokens=fused_max,
            fused_moe_ar=fused_ar,
            multimem_ok=mm,
        )

    assert pick(1) is K3MoETailTier.TAIL_FUSION
    assert pick(16) is K3MoETailTier.TAIL_FUSION
    # One past capacity must leave the fused tail rather than truncate.
    assert pick(17) is not K3MoETailTier.TAIL_FUSION
    # Outside the graph phase the fused tail is unreachable at any size.
    assert pick(1, graph_phase=False) is not K3MoETailTier.TAIL_FUSION
    # No fused tail compiled: capacity is 0 and the join tier takes decode.
    assert pick(1, fused_max=0) is K3MoETailTier.FUSED_LANE_AR
    assert pick(256) is K3MoETailTier.MULTIMEM_AR
    assert pick(8192) is K3MoETailTier.MULTIMEM_AR
    assert pick(8193) is not K3MoETailTier.MULTIMEM_AR
    assert pick(256, mm=False) is K3MoETailTier.FUSED_LANE_AR
    # fused_moe_ar reports the trtllm AR lane, which only the join tier uses.
    # The fused tail owns its own multicast collective, so it outranks the
    # flag rather than being gated by it; every other size falls to portable.
    assert pick(1, fused_ar=False) is K3MoETailTier.TAIL_FUSION
    for n in (17, 64, 256, 100_000):
        assert pick(n, fused_ar=False) is K3MoETailTier.SEPARATE_REDUCE


# 4 is included so a single node can cover the tiers; the fused tail itself
# only supports 8 and 16 and skips itself below that.
collective = pytest.mark.skipif(
    _world_size() not in {4, 8, 16},
    reason="launch with torchrun world size 4, 8 or 16",
)


@collective
@pytest.mark.parametrize("m", [1, 4, 16])
def test_fused_tail_matches_reference(m):
    """TAIL_FUSION: the decode tier, the only one with a fused kernel."""
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    if not _agreed(
        latent_tail_supported(
            tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
        )
    ):
        pytest.skip("fused latent tail unsupported here")

    tail = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test",
    )
    comm = _build_comm(dev, latent_tail=tail)
    routed, shared, prefix = _inputs(rank, dev, m, seed=11)
    ref = _reference(comm, routed, shared, prefix)
    out = comm._tail_fusion(routed, shared, prefix)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


@collective
@pytest.mark.parametrize("m", [256, 1024])
def test_multimem_ar_matches_reference(m):
    """MULTIMEM_AR: in-switch staged reduces, then the replicated projection."""
    from tokenspeed_kernel.ops.communication.multimem import multimem_available

    rank, dev = _setup()
    if not _agreed(multimem_available()):
        pytest.skip("multimem unavailable here")

    comm = _build_comm(dev)
    routed, shared, prefix = _inputs(rank, dev, m, seed=22)
    ref = _reference(comm, routed, shared, prefix)
    out = comm._tail_multimem_ar(routed, shared, prefix, m, H)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


@collective
@pytest.mark.parametrize("m", [1, MID_TOKENS, 1024])
def test_fused_lane_ar_matches_reference(m):
    """FUSED_LANE_AR: the join tier, reached by eager serving at every size.

    Driven without a lane buffer — the lane only materializes at bs==1 inside
    a captured graph, and the join's no-lane path is what eager traffic takes.
    """
    rank, dev = _setup()
    comm = _build_comm(dev)
    routed, shared, prefix = _inputs(rank, dev, m, seed=55)
    ref = _reference(comm, routed, shared, prefix)
    out = comm._tail_fused_lane_ar(routed, shared, prefix, None, m, H)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


@collective
@pytest.mark.parametrize("m", [1, MID_TOKENS, 1024])
def test_separate_reduce_matches_reference(m):
    """SEPARATE_REDUCE: the portable tier, reached whenever fused AR is off.

    No end-to-end eval in this repo selects it, so this is its only coverage.
    """
    rank, dev = _setup()
    comm = _build_comm(dev)
    routed, shared, prefix = _inputs(rank, dev, m, seed=33)
    ref = _reference(comm, routed, shared, prefix)
    out = _run_separate_reduce(comm, routed, shared, prefix, m)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.05


@collective
def test_tiers_agree_with_each_other():
    """The property that matters in serving: same input, same answer.

    A tier that is individually within tolerance of the reference can still
    sit on the opposite edge from another tier, and a request's tier depends
    on batch size — so agreement between tiers is checked directly, at one
    token count, with the selector's capacity rules bypassed.
    """
    from tokenspeed_kernel.ops.communication.multimem import multimem_available
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    m = 16  # the fused tail's capacity ceiling; every other tier accepts it

    tail = None
    if _agreed(
        latent_tail_supported(
            tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
        )
    ):
        tail = KimiK3LatentTailOp.initialize(
            group=dist.group.WORLD,
            hidden_size=H,
            latent_size=L,
            rms_eps=EPS,
            device=dev,
            layer_index=0,
            model_scope="test",
        )
    comm = _build_comm(dev, latent_tail=tail)
    routed, shared, prefix = _inputs(rank, dev, m, seed=44)

    # Every tier reduces its partials in place, so each gets its own copies;
    # feeding one tier the tensors a previous tier already reduced compares
    # a single-reduced result against a twice-reduced one.
    def fresh():
        return routed.clone(), shared.clone(), prefix.clone()

    outs = {"separate_reduce": _run_separate_reduce(comm, *fresh(), m)}
    r, s, p = fresh()
    outs["fused_lane_ar"] = comm._tail_fused_lane_ar(r, s, p, None, m, H)
    if _agreed(multimem_available()):
        outs["multimem_ar"] = comm._tail_multimem_ar(*fresh(), m, H)
    if tail is not None:
        outs["tail_fusion"] = comm._tail_fusion(*fresh())
    torch.cuda.synchronize()

    base_name, base = next(iter(outs.items()))
    scale = base.float().abs().max().item()
    for name, out in outs.items():
        diff = (out.float() - base.float()).abs().max().item() / max(scale, 1e-6)
        assert diff < 0.02, f"{name} and {base_name} disagree by {diff}"
