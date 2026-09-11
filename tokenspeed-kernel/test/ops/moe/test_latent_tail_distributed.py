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

"""Multicast latent tail: AR(latent)+norm+RS, sharded up-projection with NVLS
multicast all-gather, Lamport gather — must match the unfused reference and
survive CUDA-graph capture/replay.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-5


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {8, 16},
    reason="launch with torchrun world size 8 or 16",
)


def _require_latent_tail():
    # Collectively-agreed probe: skips must match across ranks, or the
    # remaining ranks hang in the op's rendezvous.
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
    from tokenspeed_kernel.ops.moe.latent_tail import latent_tail_supported

    # A server does this at distributed init; the gate refuses to gather it
    # itself, because it is also asked from dispatch where only a group is present.
    gather_fabric_map()
    ok = latent_tail_supported(
        tp_size=_world_size(),
        hidden_size=H,
        latent_size=L,
        dtype=torch.bfloat16,
        group=dist.group.WORLD,
    )
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        pytest.skip("platform has no rank-agreed multicast tail support")


def _setup():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, torch.device("cuda", rank)


def _reference(routed, shared, rms_w, up_w):
    lat = routed.float().clone()
    dist.all_reduce(lat)
    sh = shared.float().clone()
    dist.all_reduce(sh)
    var = lat.pow(2).mean(-1, keepdim=True)
    lat_n = (lat * torch.rsqrt(var + EPS)).to(torch.bfloat16) * rms_w
    return (lat_n.float() @ up_w.float().T + sh).to(torch.bfloat16)


def _inputs(rank, dev, m, seed):
    torch.manual_seed(1234)  # weights identical across ranks
    up_w = (torch.randn(H, L, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    rms_w = torch.randn(L, dtype=torch.bfloat16, device=dev).contiguous()
    torch.manual_seed(seed + rank)  # per-rank partials
    routed = (torch.randn(m, L, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    shared = (torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    return routed, shared, rms_w, up_w


@pytest.mark.parametrize("m", [1, 4, 5, 6, 16, 64])
def test_latent_tail_matches_reference(m):
    from tokenspeed_kernel.ops.moe.latent_tail import KimiK3LatentTailOp

    rank, dev = _setup()
    _require_latent_tail()
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test",
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, m, seed=100)
    ref = _reference(routed, shared, rms_w, up_w)
    out = op(routed, shared, rms_w, up_w)
    torch.cuda.synchronize()
    scale = ref.float().abs().max().item()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05 * max(scale, 1.0), f"m={m}: err {err} vs scale {scale}"


@pytest.mark.parametrize("m", [1, 4, 6, 16])
def test_latent_tail_graph_replay(m):
    from tokenspeed_kernel.ops.moe.latent_tail import KimiK3LatentTailOp

    rank, dev = _setup()
    _require_latent_tail()
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test",
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, m, seed=200 + m)
    for _ in range(3):
        op(routed, shared, rms_w, up_w)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(routed, shared, rms_w, up_w)
    # two replays with fresh inputs: the Lamport state must self-reset
    scale = None
    for seed in (300, 301):
        torch.manual_seed(seed + rank)
        routed.copy_(torch.randn(m, L, dtype=torch.bfloat16, device=dev) * 0.1)
        shared.copy_(torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(routed, shared, rms_w, up_w)
        scale = ref.float().abs().max().item()
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 0.05 * max(scale, 1.0), f"seed={seed}: err {err}"


@pytest.mark.parametrize("m", [1, 4, 16])
def test_latent_tail_deferred_top16_matches_materialized(m):
    from tokenspeed_kernel.ops.moe.latent_tail import KimiK3LatentTailOp

    rank, dev = _setup()
    _require_latent_tail()
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope=f"test-deferred-{m}",
        finalize_top_k=16,
    )
    _, shared, rms_w, up_w = _inputs(rank, dev, m, seed=600 + m)
    torch.manual_seed(700 + rank)
    rows = (
        torch.randn(m * 16, L, dtype=torch.bfloat16, device=dev) * 0.02
    ).contiguous()
    weights = torch.randn(m, 16, dtype=torch.bfloat16, device=dev).contiguous()
    indices = torch.arange(m * 16, dtype=torch.int32, device=dev).reshape(m, 16)
    indices[:, ::5] = -1

    routed = torch.zeros(m, L, dtype=torch.float32, device=dev)
    for slot in range(16):
        valid = indices[:, slot] >= 0
        routed[valid] += (
            weights[valid, slot, None].float()
            * rows[indices[valid, slot].long()].float()
        )
    routed = routed.to(torch.bfloat16)

    expected = op(routed, shared, rms_w, up_w).clone()
    torch.cuda.synchronize()
    dist.barrier()
    actual = op.call_deferred(
        rows,
        weights,
        indices,
        shared,
        rms_w,
        up_w,
        num_tokens=m,
    )
    torch.cuda.synchronize()
    scale = expected.float().abs().max().item()
    err = (actual.float() - expected.float()).abs().max().item()
    assert err < 0.02 * max(scale, 1.0), f"m={m}: err {err} vs scale {scale}"


@pytest.mark.parametrize("m", [9, 64])
def test_latent_tail_split_collective_multistream_graph_replay(m):
    """Prepared shared shards remain accurate and fresh across graph replays."""
    from tokenspeed_kernel.ops.moe.latent_tail import KimiK3LatentTailOp

    rank, dev = _setup()
    _require_latent_tail()
    control = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test-split-control",
    )
    split = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test-split-candidate",
        split_collective=True,
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, m, seed=400 + m)
    auxiliary = torch.cuda.Stream(device=dev)

    def split_call():
        main = torch.cuda.current_stream(dev)
        auxiliary.wait_stream(main)
        with torch.cuda.stream(auxiliary):
            prepared = split.reduce_scatter_shared(shared, rms_w)
        main.wait_stream(auxiliary)
        return split(
            routed,
            shared,
            rms_w,
            up_w,
            prepared_shared_shard=prepared,
        )

    for _ in range(3):
        split_call()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = split_call()

    for seed in range(500, 508):
        torch.manual_seed(seed + rank)
        routed.copy_(torch.randn_like(routed) * 0.1)
        shared.copy_(torch.randn_like(shared) * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        expected = control(routed, shared, rms_w, up_w)
        torch.cuda.synchronize()
        scale = expected.float().abs().max().item()
        err = (actual.float() - expected.float()).abs().max().item()
        assert err < 0.02 * max(scale, 1.0), f"seed={seed}: err {err}"


@pytest.mark.parametrize("m", [1, 4, 6, 8, 16])
def test_latent_tail_fused_prefix_matches_eager(m):
    from tokenspeed_kernel.ops.moe.latent_tail import KimiK3LatentTailOp

    rank, dev = _setup()
    _require_latent_tail()
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test",
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, m, seed=700)
    torch.manual_seed(900 + rank)
    prefix = (torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.9).contiguous()
    prefix[:, ::911] *= 8  # residual-stream outliers, as in serving
    eager = op(routed, shared, rms_w, up_w) + prefix
    # The mailbox is reused between the two calls; production separates them
    # by a whole forward, so give the sentinel cleanup the same guarantee.
    torch.cuda.synchronize()
    dist.barrier()
    fused = op(routed, shared, rms_w, up_w, prefix=prefix)
    torch.cuda.synchronize()
    assert torch.equal(eager, fused), "fused prefix add must be bit-identical"


@pytest.mark.parametrize("m", [1, 8])
def test_latent_tail_accepts_sharded_weight(m):
    """A pre-sharded ``[H/tp, L]`` weight must reproduce the replicated path.

    The column-parallel projection stores only this rank's row block; the tail
    consumes exactly that block either way, so both spellings must agree
    bitwise.
    """
    from tokenspeed_kernel.ops.moe.latent_tail import KimiK3LatentTailOp

    rank, dev = _setup()
    _require_latent_tail()
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        layer_index=0,
        model_scope="test",
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, m, seed=1100)
    shard = H // _world_size()
    up_w_shard = up_w.narrow(0, rank * shard, shard).contiguous()
    full = op(routed, shared, rms_w, up_w).clone()
    torch.cuda.synchronize()
    dist.barrier()
    sharded = op(routed, shared, rms_w, up_w_shard)
    torch.cuda.synchronize()
    assert torch.equal(full, sharded), "sharded weight must be bit-identical"


def _require_attn_collective():
    """Rank-agreed multicast probe; a one-sided skip hangs the rendezvous."""
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
    from tokenspeed_kernel.ops.moe.latent_tail import multicast_backend_available

    # The probe refuses to gather the fabric map itself; a server does it at
    # distributed init, so a standalone test has to stand in for that.
    gather_fabric_map()
    ok = multicast_backend_available(dist.group.WORLD)
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        pytest.skip("platform has no rank-agreed multicast support")


def _attn_collective(dev, max_m):
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit import (  # noqa: E501
        CollectiveKernel,
    )

    return CollectiveKernel(
        group=dist.group.WORLD,
        rank=dist.get_rank(),
        tp_size=_world_size(),
        latent_dim=H,
        hidden_dim=H,
        max_m=max_m,
        max_token_ctas=max_m,
        rms_eps=1.0,
        fp32_internal=True,
        scratch_allocator=None,
        finalize_top_k=None,
        precompile_split=True,
        residual_from_shared=True,
    )


def test_residual_from_shared_sums_the_ranks_and_adds_the_prefix():
    """The attention epilogue must emit all-reduce(partial) + prefix.

    One instance, many widths: production reuses a single kernel across ~93
    calls per step, so the Lamport re-arm between calls is part of what the
    epilogue has to get right.
    """
    rank, dev = _setup()
    _require_attn_collective()
    max_m = 8
    kernel = _attn_collective(dev, max_m)
    # Not ones: a gamma-scaled or RMS-normalised epilogue must not pass here.
    gamma = torch.full((H,), 3.0, dtype=torch.bfloat16, device=dev).contiguous()

    for call, m in enumerate((1, 8, 1, 3, 8, 2, 5, 1)):
        torch.manual_seed(7 + rank + 31 * call)
        partial = (
            torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.1
        ).contiguous()
        torch.manual_seed(99 + call)  # rank-uniform, like the residual stream
        prefix = (
            torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.1
        ).contiguous()

        expected = partial.float().clone()
        dist.all_reduce(expected)
        expected = (expected + prefix.float()).to(torch.bfloat16)

        out, _ = kernel(
            partial,
            prefix,
            gamma,
            include_reduce_scatter=False,
            include_routed=True,
            latent_output_override=torch.empty(
                max_m, H, dtype=torch.bfloat16, device=dev
            ),
        )
        torch.cuda.synchronize()
        err = (out.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item()
        tol = 8e-3 * max(scale, 1.0)  # one bf16 ulp; reduce order differs
        assert err <= tol, f"call={call} m={m} max|err|={err} scale={scale}"

        # Negative control: the same comparison must reject a missing residual,
        # or it cannot tell a correct epilogue from one that drops the prefix.
        assert (
            out.float() - prefix.float() - expected.float()
        ).abs().max().item() > tol


def test_residual_from_shared_writes_the_caller_s_buffer():
    """The returned rows must live in the override, not the instance scratch.

    Reading the returned tensor alone cannot tell the two apart -- both are
    ``latent_output[:m]`` -- so compare storage, and call twice so a kernel
    that honours the override once but not again is caught.
    """
    _, dev = _setup()
    _require_attn_collective()
    max_m, m = 8, 2
    kernel = _attn_collective(dev, max_m)
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()
    partial = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    prefix = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()

    seen = []
    for _ in range(2):
        override = torch.empty(max_m, H, dtype=torch.bfloat16, device=dev)
        out, _ = kernel(
            partial,
            prefix,
            gamma,
            include_reduce_scatter=False,
            include_routed=True,
            latent_output_override=override,
        )
        torch.cuda.synchronize()
        assert out.data_ptr() == override.data_ptr()
        seen.append(override.data_ptr())
    # Positive control: the two calls really did hand in different buffers, so
    # the equality above is not trivially satisfied by one reused allocation.
    assert seen[0] != seen[1]


@pytest.mark.parametrize(
    "bad",
    ["rows", "dtype", "strided"],
)
def test_residual_from_shared_rejects_an_unusable_override(bad):
    """The dispatch bakes a static layout, so a mis-shaped buffer must not run."""
    _, dev = _setup()
    _require_attn_collective()
    max_m, m = 8, 1
    kernel = _attn_collective(dev, max_m)
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()
    partial = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    prefix = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    if bad == "rows":
        override = torch.empty(max_m - 1, H, dtype=torch.bfloat16, device=dev)
    elif bad == "dtype":
        override = torch.empty(max_m, H, dtype=torch.float16, device=dev)
    else:
        override = torch.empty(max_m, 2 * H, dtype=torch.bfloat16, device=dev)[:, ::2]
    with pytest.raises(ValueError, match="latent output buffer"):
        kernel(
            partial,
            prefix,
            gamma,
            include_reduce_scatter=False,
            include_routed=True,
            latent_output_override=override,
        )


def test_residual_from_shared_rejects_the_reduce_scatter_role():
    """This epilogue emits reduced+residual; scattering the residual is nonsense."""
    _, dev = _setup()
    _require_attn_collective()
    m = 2
    kernel = _attn_collective(dev, 8)
    partial = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    prefix = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()
    override = torch.empty(8, H, dtype=torch.bfloat16, device=dev)
    with pytest.raises(ValueError, match="include_reduce_scatter"):
        kernel(
            partial,
            prefix,
            gamma,
            include_reduce_scatter=True,
            include_routed=True,
            latent_output_override=override,
        )
    # Positive control: the same call with the role off must go through, or the
    # test would pass on a kernel that rejects everything.
    out, _ = kernel(
        partial,
        prefix,
        gamma,
        include_reduce_scatter=False,
        include_routed=True,
        latent_output_override=override,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, H)


def test_residual_from_shared_rejects_a_mismatched_latent_width():
    """The residual read walks shared_source with the latent pitch."""
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit import (  # noqa: E501
        CollectiveKernel,
    )

    _setup()
    with pytest.raises(ValueError, match="latent_dim == hidden_dim"):
        CollectiveKernel(
            group=dist.group.WORLD,
            rank=dist.get_rank(),
            tp_size=_world_size(),
            latent_dim=L,
            hidden_dim=H,
            max_m=8,
            max_token_ctas=8,
            rms_eps=1.0,
            fp32_internal=True,
            scratch_allocator=None,
            finalize_top_k=None,
            precompile_split=True,
            residual_from_shared=True,
        )
