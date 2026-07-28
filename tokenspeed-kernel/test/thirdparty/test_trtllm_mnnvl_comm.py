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
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, torch.device("cuda", rank)


_workspaces: dict = {}


def _get_workspaces():
    """Create (once) both the IPC lamport and the MNNVL workspaces."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_mnnvl_workspace_for_all_reduce_fusion,
    )

    if not _workspaces:
        rank, dev = _setup()
        world = dist.get_world_size()
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


def test_latent_norm_matches_ipc_backend():
    from tokenspeed_kernel.thirdparty.cuda.trtllm import AllReduceFusionPattern

    ws = _skip_unless_mnnvl()
    rank, dev = ws["rank"], ws["dev"]
    token_num, lane_dim = 2, L + H
    torch.manual_seed(400 + rank)
    torch.manual_seed(7)
    gamma = torch.randn(L, dtype=torch.bfloat16, device=dev).contiguous()

    results = {}
    for label in ("ipc", "mnnvl"):
        torch.manual_seed(400 + rank)
        lane = (
            torch.randn(token_num, lane_dim, dtype=torch.bfloat16, device=dev) * 0.1
        ).contiguous()
        _run_ar(
            ws[label],
            lane,
            lane,
            dict(
                pattern_code=AllReduceFusionPattern.kAllReduceLatentNorm,
                allreduce_out=lane,
                rms_gamma=gamma,
                rms_eps=EPS,
                latent_width=L,
            ),
            token_num,
            lane_dim,
        )
        torch.cuda.synchronize()
        results[label] = lane
        dist.barrier()

    torch.testing.assert_close(results["ipc"], results["mnnvl"], atol=1e-3, rtol=1e-3)


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
