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

"""MNNVL-structured one-shot AR fusion: NVLS multicast payload store +
Lamport rotation, with the vendored FusedOp epilogues (incl. the Kimi-K3
patterns kARResidualAttnResCombine and kAllReduceLatentNorm). Must match the
IPC lamport backend and survive CUDA-graph capture/replay (the rotation state
lives in device memory and self-resets across replays).

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-6
MAXTOK = 32


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {2, 4, 8},
    reason="launch with torchrun world size 2, 4 or 8",
)


def _setup():
    lrank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lrank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    # dist.group.WORLD workspaces need the GLOBAL rank: LOCAL_RANK repeats on
    # every node, so using it multi-node hands two ranks the same workspace ID.
    return dist.get_rank(), torch.device("cuda", lrank)


def _spans_nodes() -> bool:
    """True when the world spans hosts. Cross-node CUDA-IPC workspace creation
    fails AND poisons the CUDA context ('invalid resource handle' on the next
    allocation), so it must be skipped outright rather than caught."""
    import socket

    names = [None] * dist.get_world_size()
    dist.all_gather_object(names, socket.gethostname())
    return len(set(names)) > 1


_workspaces: dict = {}


def _get_workspaces():
    """Create (once) the MNNVL workspace, plus IPC when single-node.

    Cross-node ``ipc`` is None -- tests comparing against the IPC backend skip
    individually; mnnvl-only tests still run.
    """
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_mnnvl_workspace_for_all_reduce_fusion,
    )

    if not _workspaces:
        rank, dev = _setup()
        world = dist.get_world_size()
        ipc_ws = None
        if not _spans_nodes():
            _, ipc_ws = trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world, MAXTOK, H + L, group=dist.group.WORLD
            )
        mnnvl_ws = trtllm_create_mnnvl_workspace_for_all_reduce_fusion(
            rank, world, MAXTOK, H + L, group=dist.group.WORLD
        )
        _workspaces.update(rank=rank, dev=dev, world=world, ipc=ipc_ws, mnnvl=mnnvl_ws)
    return _workspaces


def _skip_unless_mnnvl():
    try:
        ws = _get_workspaces()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mnnvl workspace unavailable: {exc}")
    return ws


def _skip_unless_ipc(ws):
    if ws["ipc"] is None:
        pytest.skip("IPC lamport workspace unavailable (world spans nodes)")


def _run_ar(ws, x, out, pattern_kwargs, token_num, hidden_dim):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import trtllm_allreduce_fusion

    c = _workspaces
    trtllm_allreduce_fusion(
        allreduce_in=x,
        world_size=c["world"],
        world_rank=c["rank"],
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=ws,
        launch_with_pdl=False,
        trigger_completion_at_end=True,
        fp32_acc=False,
        use_oneshot=True,
        **pattern_kwargs,
    )
    return out


@pytest.mark.parametrize("token_num", [1, 4, 32])
def test_plain_allreduce_matches_nccl(token_num):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ws = _skip_unless_mnnvl()
    rank, dev = ws["rank"], ws["dev"]
    torch.manual_seed(100 + rank)
    x = (torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    out = torch.empty_like(x)
    _run_ar(
        ws["mnnvl"],
        x,
        out,
        dict(pattern_code=AllReduceFusionPattern.kAllReduce, allreduce_out=out),
        token_num,
        H,
    )
    torch.cuda.synchronize()
    ref = x.float().clone()
    dist.all_reduce(ref)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("token_num", [1, 8])
def test_residual_rmsnorm_matches_ipc_backend(token_num):
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ws = _skip_unless_mnnvl()
    _skip_unless_ipc(ws)
    rank, dev = ws["rank"], ws["dev"]
    torch.manual_seed(200 + rank)
    x = (torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    residual = (
        torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
    ).contiguous()
    torch.manual_seed(7)
    gamma = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()

    results = {}
    for label in ("ipc", "mnnvl"):
        norm_out = torch.empty_like(x)
        res_out = torch.empty_like(x)
        _run_ar(
            ws[label],
            x,
            norm_out,
            dict(
                pattern_code=AllReduceFusionPattern.kARResidualRMSNorm,
                residual_in=residual,
                residual_out=res_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=EPS,
            ),
            token_num,
            H,
        )
        torch.cuda.synchronize()
        results[label] = (norm_out, res_out)
        dist.barrier()

    # Same deterministic rank-order bf16 reduction + identical FusedOp
    # epilogue: outputs should agree to bf16 rounding.
    for a, b in zip(results["ipc"], results["mnnvl"]):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_attnres_combine_matches_ipc_backend():
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ws = _skip_unless_mnnvl()
    _skip_unless_ipc(ws)
    rank, dev = ws["rank"], ws["dev"]
    token_num = 2
    torch.manual_seed(300 + rank)
    x = (torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    residual = (
        torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
    ).contiguous()
    torch.manual_seed(7)
    gamma = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()
    res_w = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()
    out_w = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()
    torch.manual_seed(9)
    sc_m = torch.randn(token_num, dtype=torch.float32, device=dev).abs().contiguous()
    sc_s = (
        torch.randn(token_num, dtype=torch.float32, device=dev).abs() + 1.0
    ).contiguous()
    sc_acc = torch.randn(token_num, H, dtype=torch.float32, device=dev).contiguous()

    results = {}
    for label in ("ipc", "mnnvl"):
        norm_out = torch.empty_like(x)
        res_out = torch.empty_like(x)
        _run_ar(
            ws[label],
            x,
            norm_out,
            dict(
                pattern_code=AllReduceFusionPattern.kARResidualAttnResCombine,
                residual_in=residual,
                residual_out=res_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=EPS,
                attnres_m=sc_m,
                attnres_s=sc_s,
                attnres_acc=sc_acc,
                attnres_res_w=res_w,
                attnres_out_norm_w=out_w,
            ),
            token_num,
            H,
        )
        torch.cuda.synchronize()
        results[label] = (norm_out, res_out)
        dist.barrier()

    for a, b in zip(results["ipc"], results["mnnvl"]):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_latent_norm_supported_by_mnnvl():
    """kAllReduceLatentNorm IS served by the mnnvl kernel.

    Cross-node there is no IPC workspace, and NCCL is ~3.3x slower, so the
    pattern must run on mnnvl -- excluding it crashed K3's latent-MoE tail with
    no fallback. supports() therefore returns True. The single-node *preference*
    for the IPC lamport workspace on this wide [latent|hidden] lane (measured
    8.60us vs 6.64 on 8x B300) is enforced in _ar_fusion_workspace when an IPC
    workspace exists, not by supports()."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ws = _skip_unless_mnnvl()
    assert ws["mnnvl"].supports(
        2,
        L + H,
        torch.bfloat16,
        ws["world"],
        AllReduceFusionPattern.kAllReduceLatentNorm,
        use_oneshot=True,
    )


def test_graph_replay_self_reset():
    """The rotation state must self-reset: capture several AR calls in one
    graph and replay it many times; every replay must produce the correct
    result for freshly copied inputs."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ws = _skip_unless_mnnvl()
    rank, dev = ws["rank"], ws["dev"]
    token_num = 1
    x = torch.zeros(token_num, H, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(x)

    def call():
        _run_ar(
            ws["mnnvl"],
            x,
            out,
            dict(pattern_code=AllReduceFusionPattern.kAllReduce, allreduce_out=out),
            token_num,
            H,
        )

    # warmup outside capture
    x.normal_()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    dist.barrier()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):  # multiple rotations per replay
            call()
    torch.cuda.synchronize()
    dist.barrier()

    for it in range(50):
        torch.manual_seed(1000 + 17 * it + rank)
        src = torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
        x.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        ref = src.float().clone()
        dist.all_reduce(ref)
        torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)
        dist.barrier()


def test_rmsnorm_family_unfused_fallback():
    """Patterns the mnnvl kernel cannot serve degrade to the unfused NCCL path.

    Cross-node there is no IPC workspace, so block-quant / partial-out rmsnorm
    calls used to raise out of the forward pass. They now run an unfused
    NCCL all-reduce + torch epilogue with the same return contract. Verified
    against the fp32 ground-truth epilogue; residual_reduce_scattered (whose
    input arrives pre-scattered) must still raise rather than corrupt.
    """
    from tokenspeed_kernel.ops.communication import trtllm as comm

    ws = _skip_unless_mnnvl()
    rank, world, dev = ws["rank"], ws["world"], _workspaces["dev"]
    pg = dist.group.WORLD
    assert comm.ensure_workspace_initialized(
        rank=rank, group=pg, max_token_num=MAXTOK, hidden_dim=H
    )
    mgr = comm._workspace_manager

    token_num = 6
    torch.manual_seed(9_000 + rank)
    x = (torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    res = torch.randn(token_num, H, dtype=torch.bfloat16, device=dev) * 0.1
    w = torch.randn(H, dtype=torch.bfloat16, device=dev).abs().contiguous()
    dist.broadcast(res, src=0)
    dist.broadcast(w, src=0)
    res = res.contiguous()

    saved = mgr.workspace_tensor, mgr.mnnvl_workspace
    mgr.workspace_tensor = None
    mgr.mnnvl_workspace = None
    try:
        norm_out, residual_out, scale_out, partial = comm.allreduce_residual_rmsnorm(
            input_tensor=x.clone(),
            residual=res,
            weight=w,
            rank=rank,
            group=pg,
            eps=EPS,
            max_token_num=MAXTOK,
        )
        with pytest.raises(RuntimeError):
            comm.allreduce_residual_rmsnorm(
                input_tensor=x.clone(),
                residual=res,
                weight=w,
                rank=rank,
                group=pg,
                eps=EPS,
                max_token_num=MAXTOK,
                residual_reduce_scattered=True,
            )
    finally:
        mgr.workspace_tensor, mgr.mnnvl_workspace = saved

    assert scale_out is None and partial is None
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x)
    gt32 = torch.stack(parts).float().sum(0) + res.float()
    gtn = gt32 * torch.rsqrt(gt32.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    scale = gtn.abs().max().item() or 1.0
    assert (residual_out.float() - gt32).abs().max().item() <= 2e-2
    assert (norm_out.float() - gtn).abs().max().item() <= 0.02 * scale + 0.01
    # replicated contract: identical on every rank
    g = [torch.empty_like(norm_out) for _ in range(world)]
    dist.all_gather(g, norm_out)
    assert all(torch.equal(g[0], gi) for gi in g)
