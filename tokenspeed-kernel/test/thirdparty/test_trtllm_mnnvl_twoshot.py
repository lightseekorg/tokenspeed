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

"""Two-shot MNNVL AR fusion: token-sharded reduce (owner reduces its shard and
multicasts the pre-epilogue sum; every rank re-runs the deterministic epilogue).

Covers what the one-shot suite cannot: token counts above the byte-based
one-shot threshold, the one-shot/two-shot dispatch boundary, agreement
between the two paths, cross-rank bitwise identity of the epilogue, repeated
launches (Lamport 3-slot rotation), and CUDA-graph capture/replay.

Single-GPU pytest runs skip this file. Exercise it with::

    torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>

or across nodes with the usual RANK/LOCAL_RANK/WORLD_SIZE env (the two-shot
path is specifically a multi-node concern).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, EPS = 7168, 1e-6
# Covers the full two-shot range and forced one-shot comparisons.
MAXTOK = 2048


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _supported_world_sizes() -> tuple[int, ...]:
    """Derive the guard from the kernel's own list so the two cannot drift:
    adding a world size to the kernel must not leave these tests silently
    skipped (which reads as success -- pytest exits 0 either way)."""
    try:
        from tokenspeed_kernel.thirdparty.cuda.trtllm import (
            _MNNVL_SUPPORTED_WORLD_SIZES,
        )

        return tuple(_MNNVL_SUPPORTED_WORLD_SIZES)
    except Exception:  # noqa: BLE001 -- kernel package unavailable
        return (2, 4, 8)


pytestmark = pytest.mark.skipif(
    _world_size() not in _supported_world_sizes(),
    reason=f"launch with torchrun world size in {_supported_world_sizes()}",
)


def _spans_nodes() -> bool:
    """True when the world spans hosts. Cross-node CUDA-IPC workspace creation
    fails AND poisons the CUDA context ('invalid resource handle' on the next
    allocation), so it must be skipped outright rather than caught."""
    import socket

    names = [None] * dist.get_world_size()
    dist.all_gather_object(names, socket.gethostname())
    return len(set(names)) > 1


_ctx: dict = {}


def _get_ctx():
    """Create (once) the IPC lamport and MNNVL workspaces sized for MAXTOK."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_mnnvl_workspace_for_all_reduce_fusion,
    )

    if not _ctx:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        dev = torch.device("cuda", rank)
        world = dist.get_world_size()
        grank = dist.get_rank()
        # mnnvl FIRST: it is the path under test, and a failed cross-node IPC
        # attempt would poison the CUDA context before we ever get here.
        mnnvl_ws = trtllm_create_mnnvl_workspace_for_all_reduce_fusion(
            grank, world, MAXTOK, H, group=dist.group.WORLD
        )
        ipc_ws = None
        if not _spans_nodes():
            _, ipc_ws = trtllm_create_ipc_workspace_for_all_reduce_fusion(
                grank, world, MAXTOK, H, group=dist.group.WORLD
            )
        _ctx.update(rank=grank, dev=dev, world=world, ipc=ipc_ws, mnnvl=mnnvl_ws)
    return _ctx


def _skip_unless_mnnvl():
    try:
        return _get_ctx()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mnnvl workspace unavailable: {exc}")


def _ar(ws, x, token_num, *, use_oneshot, pattern_kwargs, pattern_code):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import trtllm_allreduce_fusion

    c = _ctx
    trtllm_allreduce_fusion(
        allreduce_in=x,
        world_size=c["world"],
        world_rank=c["rank"],
        token_num=token_num,
        hidden_dim=H,
        workspace_ptrs=ws,
        launch_with_pdl=False,
        trigger_completion_at_end=True,
        fp32_acc=False,
        use_oneshot=use_oneshot,
        pattern_code=pattern_code,
        **pattern_kwargs,
    )


def _inputs(token_num, dev, rank, seed=0):
    torch.manual_seed(1000 + seed * 97 + rank)
    return (
        torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
    ).contiguous()


def _ref_allreduce(x):
    ref = x.float().clone()
    dist.all_reduce(ref)
    return ref


# --------------------------------------------------------------------------
# plain all-reduce
# --------------------------------------------------------------------------
@pytest.mark.parametrize("token_num", [129, 256, 1024, 2048])
def test_twoshot_plain_allreduce_matches_nccl(token_num):
    """Above the one-shot traffic threshold, two-shot must still match NCCL."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ctx = _skip_unless_mnnvl()
    x = _inputs(token_num, ctx["dev"], ctx["rank"])
    out = torch.empty_like(x)
    _ar(
        ctx["mnnvl"],
        x,
        token_num,
        use_oneshot=False,
        pattern_code=AllReduceFusionPattern.kAllReduce,
        pattern_kwargs=dict(allreduce_out=out),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), _ref_allreduce(x), atol=3e-2, rtol=3e-2)


def test_dispatch_boundary_oneshot_vs_twoshot():
    """Shapes on either side of the traffic boundary match NCCL."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        MNNVL_ONESHOT_BYTES,
        AllReduceFusionPattern,
        _mnnvl_should_use_oneshot,
    )

    ctx = _skip_unless_mnnvl()
    boundary = MNNVL_ONESHOT_BYTES // (H * ctx["world"] * 2)
    assert boundary > 0
    for token_num in (boundary, boundary + 1):
        use_oneshot = _mnnvl_should_use_oneshot(token_num, H, 2, ctx["world"])
        assert use_oneshot == (token_num == boundary)
        x = _inputs(token_num, ctx["dev"], ctx["rank"], seed=3)
        out = torch.empty_like(x)
        _ar(
            ctx["mnnvl"],
            x,
            token_num,
            use_oneshot=use_oneshot,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            pattern_kwargs=dict(allreduce_out=out),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), _ref_allreduce(x), atol=3e-2, rtol=3e-2)


def test_forced_strategies_agree_at_same_token_count():
    """The device must honor use_oneshot instead of recomputing dispatch."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ctx = _skip_unless_mnnvl()
    token_num = min(4, ctx["mnnvl"].oneshot_token_cap)
    assert token_num > 0
    x = _inputs(token_num, ctx["dev"], ctx["rank"], seed=5)
    outputs = []
    recorded_stages = []
    for use_oneshot in (True, False):
        out = torch.empty_like(x)
        _ar(
            ctx["mnnvl"],
            x,
            token_num,
            use_oneshot=use_oneshot,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            pattern_kwargs=dict(allreduce_out=out),
        )
        torch.cuda.synchronize()
        outputs.append(out)
        recorded_stages.append(int(ctx["mnnvl"].buffer_flags[3].item()))
    assert recorded_stages == [1, 2]
    torch.testing.assert_close(outputs[0], outputs[1], atol=3e-2, rtol=3e-2)


def test_workspace_dispatch_cap_is_frozen(monkeypatch):
    """A later environment change cannot enlarge an existing one-shot lane."""
    ctx = _skip_unless_mnnvl()
    workspace = ctx["mnnvl"]
    monkeypatch.setenv("TOKENSPEED_MNNVL_ONESHOT_BYTES", str(1 << 40))
    assert workspace.resolve_use_oneshot(workspace.oneshot_token_cap, None)
    assert not workspace.resolve_use_oneshot(workspace.oneshot_token_cap + 1, None)
    assert not workspace.resolve_use_oneshot(workspace.oneshot_token_cap + 1, True)


# --------------------------------------------------------------------------
# fused residual + rmsnorm (the pattern prefill actually uses)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("token_num", [129, 512, 2048])
def test_twoshot_residual_rmsnorm_matches_reference(token_num):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ctx = _skip_unless_mnnvl()
    dev, rank = ctx["dev"], ctx["rank"]
    x = _inputs(token_num, dev, rank, seed=7)
    torch.manual_seed(4242)  # residual/weight identical on every rank
    residual = (
        torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
    ).contiguous()
    weight = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    _ar(
        ctx["mnnvl"],
        x,
        token_num,
        use_oneshot=False,
        pattern_code=AllReduceFusionPattern.kARResidualRMSNorm,
        pattern_kwargs=dict(
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=weight,
            rms_eps=EPS,
        ),
    )
    torch.cuda.synchronize()
    ref_sum = _ref_allreduce(x)
    ref_residual = ref_sum + residual.float()
    var = ref_residual.pow(2).mean(-1, keepdim=True)
    ref_norm = ref_residual * torch.rsqrt(var + EPS) * weight.float()
    torch.testing.assert_close(residual_out.float(), ref_residual, atol=5e-2, rtol=5e-2)
    # The kernel accumulates in bf16 (~0.8% relative) while the reference sums in
    # fp32, and the norm scales that error by gamma (randn, |gamma| up to ~4), so
    # the absolute bound has to grow with both the world size and the gamma range.
    # Taking a max over millions of elements samples the far tail: at world 16 a
    # single element in 14.7M exceeded a flat 5e-2 while every other check --
    # residual_out, and bitwise agreement across ranks -- passed. A real kernel
    # fault corrupts whole tokens or lanes, not one isolated element.
    norm_atol = 5e-2 * max(1.0, (ctx["world"] / 8.0) ** 0.5) * float(weight.abs().max())
    torch.testing.assert_close(norm_out.float(), ref_norm, atol=norm_atol, rtol=5e-2)


def test_twoshot_epilogue_bitwise_identical_across_ranks():
    """Every rank re-runs the epilogue locally; outputs must match bit-for-bit,
    otherwise ranks silently diverge after a few layers."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ctx = _skip_unless_mnnvl()
    dev, world = ctx["dev"], ctx["world"]
    token_num = 512
    x = _inputs(token_num, dev, ctx["rank"], seed=11)
    torch.manual_seed(999)
    residual = (
        torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
    ).contiguous()
    weight = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    _ar(
        ctx["mnnvl"],
        x,
        token_num,
        use_oneshot=False,
        pattern_code=AllReduceFusionPattern.kARResidualRMSNorm,
        pattern_kwargs=dict(
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=weight,
            rms_eps=EPS,
        ),
    )
    torch.cuda.synchronize()
    gathered = [torch.empty_like(norm_out) for _ in range(world)]
    dist.all_gather(gathered, norm_out)
    for peer, got in enumerate(gathered):
        assert torch.equal(got, gathered[0]), f"rank {peer} epilogue differs bitwise"


# --------------------------------------------------------------------------
# rotation state and graph capture
# --------------------------------------------------------------------------
def test_twoshot_repeated_launches_rotate_cleanly():
    """The Lamport 3-slot rotation must survive back-to-back launches; a stale
    sentinel would hang or corrupt the 4th call."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ctx = _skip_unless_mnnvl()
    token_num = 300
    for i in range(6):
        x = _inputs(token_num, ctx["dev"], ctx["rank"], seed=20 + i)
        out = torch.empty_like(x)
        _ar(
            ctx["mnnvl"],
            x,
            token_num,
            use_oneshot=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            pattern_kwargs=dict(allreduce_out=out),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), _ref_allreduce(x), atol=3e-2, rtol=3e-2, msg=f"iteration {i}"
        )


def test_twoshot_cuda_graph_capture_replay():
    """Prefill graphs replay the kernel; rotation state lives in device memory
    and must self-advance across replays."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ctx = _skip_unless_mnnvl()
    token_num = 256
    x = _inputs(token_num, ctx["dev"], ctx["rank"], seed=31)
    out = torch.empty_like(x)

    def launch():
        _ar(
            ctx["mnnvl"],
            x,
            token_num,
            use_oneshot=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            pattern_kwargs=dict(allreduce_out=out),
        )

    launch()  # warm up before capture
    torch.cuda.synchronize()
    dist.barrier()

    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.cuda.graph(g):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    ref = _ref_allreduce(x)
    for i in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), ref, atol=3e-2, rtol=3e-2, msg=f"replay {i}"
        )
