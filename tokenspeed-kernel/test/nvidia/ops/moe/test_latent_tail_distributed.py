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

"""Distributed contracts for the retained attention all-reduce epilogue."""

import os

import pytest
import torch
import torch.distributed as dist

H, L = 7168, 3584


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {8, 16}, reason="requires TP8 or TP16"
)


def _setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), torch.device("cuda", local_rank)


def _require_attn_collective():
    """Rank-agreed multicast probe; a one-sided skip hangs the rendezvous."""
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
    from tokenspeed_kernel.ops.moe.latent_tail import multicast_backend_available

    # The probe will not gather the fabric map; a server does it at init.
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
        )
        torch.cuda.synchronize()
        err = (out.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item()
        tol = 8e-3 * max(scale, 1.0)  # one bf16 ulp; reduce order differs
        assert err <= tol, f"call={call} m={m} max|err|={err} scale={scale}"

        # Negative control: the comparison must reject a dropped residual.
        assert (
            out.float() - prefix.float() - expected.float()
        ).abs().max().item() > tol


def test_residual_from_shared_rejects_the_reduce_scatter_role():
    """This epilogue emits reduced+residual; scattering the residual is nonsense."""
    _, dev = _setup()
    _require_attn_collective()
    m = 2
    kernel = _attn_collective(dev, 8)
    partial = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    prefix = torch.zeros(m, H, dtype=torch.bfloat16, device=dev).contiguous()
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()
    with pytest.raises(ValueError, match="include_reduce_scatter"):
        kernel(
            partial,
            prefix,
            gamma,
            include_reduce_scatter=True,
            include_routed=True,
        )
    # Positive control: a kernel that rejected everything would pass the line above.
    out, _ = kernel(
        partial,
        prefix,
        gamma,
        include_reduce_scatter=False,
        include_routed=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, H)


def test_residual_from_shared_chains_through_its_own_latent_buffer():
    """Layer L's result is layer L+1's residual, and it lives in the kernel.

    The caller keeps no buffer of its own, so each call reads the buffer the
    previous call wrote and writes it again. The epilogue loads and stores one
    element per thread at one offset, which is what makes that legal; this
    pins it against a chain long enough to catch a generation-rotation bug.
    """
    rank, dev = _setup()
    _require_attn_collective()
    depth, m = 24, 2
    kernel = _attn_collective(dev, 8)
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()

    def partial_for(layer):
        g = torch.Generator(device=dev).manual_seed(1000 * layer + rank)
        return (
            (torch.randn(m, H, generator=g, dtype=torch.float32, device=dev) * 0.1)
            .to(torch.bfloat16)
            .contiguous()
        )

    g0 = torch.Generator(device=dev).manual_seed(7)  # residual is rank-uniform
    x = (
        (torch.randn(m, H, generator=g0, dtype=torch.float32, device=dev) * 0.1)
        .to(torch.bfloat16)
        .contiguous()
    )
    expected = x.float()

    last_acc = None
    seen_ptrs = []
    for layer in range(depth):
        part = partial_for(layer)
        acc = part.float().clone()
        dist.all_reduce(acc)
        last_acc = acc
        expected = (expected + acc).to(torch.bfloat16).float()
        x, _ = kernel(part, x, gamma, include_reduce_scatter=False, include_routed=True)
        seen_ptrs.append(x.data_ptr())
        torch.cuda.synchronize()

    # Values alone cannot tell reuse from a fresh buffer per call.
    assert len(set(seen_ptrs)) == 1, f"buffer moved across calls: {seen_ptrs}"
    err = (x.float() - expected).abs().max().item()
    scale = expected.abs().max().item()
    # The steps are dependent and the reduce orders differ: budget the walk.
    tol = 8e-3 * max(scale, 1.0) * depth**0.5
    assert err <= tol, f"depth={depth} max|err|={err} scale={scale} tol={tol}"
    # Negative control: a chain missing its last layer must fail this.
    assert (x.float() - (expected - last_acc)).abs().max().item() > tol


def test_residual_from_shared_chains_inside_a_cuda_graph():
    """Production runs this chain inside the decode graph, with no host sync.

    The chained version above synchronizes between calls, so on its own it
    cannot tell a working chain from one the host happened to serialize.
    Capture the chain instead and replay it.
    """
    rank, dev = _setup()
    _require_attn_collective()
    depth, m = 12, 8
    kernel = _attn_collective(dev, 8)
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()

    def partial_for(layer):
        g = torch.Generator(device=dev).manual_seed(4000 + 7 * layer + rank)
        return (
            (torch.randn(m, H, generator=g, dtype=torch.float32, device=dev) * 0.1)
            .to(torch.bfloat16)
            .contiguous()
        )

    parts = [partial_for(i) for i in range(depth)]
    g0 = torch.Generator(device=dev).manual_seed(11)
    seed_residual = (
        (torch.randn(m, H, generator=g0, dtype=torch.float32, device=dev) * 0.1)
        .to(torch.bfloat16)
        .contiguous()
    )

    def chain():
        x = seed_residual
        for part in parts:
            x, _ = kernel(
                part, x, gamma, include_reduce_scatter=False, include_routed=True
            )
        return x

    eager = chain().clone()  # also warms the Lamport state before capture
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = chain()

    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        # Bit-exact: the buffer rotation must land in the same place every replay.
        assert torch.equal(captured, eager)

    # Positive control: perturbing an input must move the captured chain.
    parts[0].copy_(parts[0] * 2)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(captured, eager)


def test_residual_from_shared_refuses_a_width_past_its_buffer():
    """With no caller buffer left to validate, this bound is the only guard.

    The internal latent output is exactly max_m rows, so an over-wide launch
    would write past it.
    """
    _, dev = _setup()
    _require_attn_collective()
    max_m = 8
    kernel = _attn_collective(dev, max_m)
    gamma = torch.ones(H, dtype=torch.bfloat16, device=dev).contiguous()
    for bad_m in (max_m + 1, max_m * 2):
        part = torch.zeros(bad_m, H, dtype=torch.bfloat16, device=dev).contiguous()
        with pytest.raises(ValueError, match="runtime M"):
            kernel(
                part,
                part,
                gamma,
                include_reduce_scatter=False,
                include_routed=True,
            )
    # Positive control: a bound that rejected everything would pass the loop above.
    ok = torch.zeros(max_m, H, dtype=torch.bfloat16, device=dev).contiguous()
    out, _ = kernel(ok, ok, gamma, include_reduce_scatter=False, include_routed=True)
    torch.cuda.synchronize()
    assert out.shape == (max_m, H)


def test_residual_from_shared_precompiles_the_variant_it_dispatches():
    """Lazy compilation inside a graph capture is what precompile prevents.

    Narrowing the variant list to the callable one is only safe if that one
    is still compiled up front; ``launch`` would otherwise JIT on first use.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit import (  # noqa: E501
        _COMPILED,
        _compile_key,
    )

    _, dev = _setup()
    _require_attn_collective()
    max_m = 8
    _attn_collective(dev, max_m)
    key = _compile_key(
        rank=dist.get_rank(),
        tp_size=_world_size(),
        latent_dim=H,
        hidden_dim=H,
        max_m=max_m,
        max_token_ctas=max_m,
        fp32_internal=True,
        residual_from_shared=True,
        include_reduce_scatter=False,
        include_routed=True,
        finalize_top_k=None,
    )
    assert key in _COMPILED
    # Positive control: without this, membership above would say nothing.
    other = _compile_key(
        rank=dist.get_rank(),
        tp_size=_world_size(),
        latent_dim=H,
        hidden_dim=H,
        max_m=max_m,
        max_token_ctas=max_m,
        fp32_internal=True,
        residual_from_shared=True,
        include_reduce_scatter=True,
        include_routed=True,
        finalize_top_k=None,
    )
    assert other not in _COMPILED


def test_residual_from_shared_rejects_a_build_it_could_never_dispatch():
    """Routed-only is a split dispatch, and a split dispatch needs the compile.

    Without the constructor check this builds, rendezvouses symmetric memory,
    and only then raises -- on every call, forever.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit import (  # noqa: E501
        CollectiveKernel,
    )

    _setup()
    with pytest.raises(ValueError, match="precompile_split"):
        CollectiveKernel(
            group=dist.group.WORLD,
            rank=dist.get_rank(),
            tp_size=_world_size(),
            latent_dim=H,
            hidden_dim=H,
            max_m=8,
            max_token_ctas=8,
            rms_eps=1.0,
            fp32_internal=True,
            scratch_allocator=None,
            finalize_top_k=None,
            precompile_split=False,
            residual_from_shared=True,
        )


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
