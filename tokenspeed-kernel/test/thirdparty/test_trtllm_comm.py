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

"""
Multi-GPU tests for tokenspeed_kernel.ops.communication.

Correctness check: approximate match against torch reference (IPC-based reduction differs
from NCCL reduction order, so bf16 results may differ slightly -- this is expected).

Run with:
    pytest tokenspeed-kernel/test/thirdparty/test_trtllm_comm.py -v
"""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="trtllm comm kernels are NVIDIA-only",
)

_num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

requires_multi_gpu = pytest.mark.skipif(_num_gpus < 2, reason="need >=2 GPUs")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(rank, world_size, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    return device


def cleanup():
    dist.destroy_process_group()


# ─── Test: Pattern enum values ───

# Expected enum values — must match the C++ kernel definitions.
EXPECTED_ENUMS = {
    "AllReduceFusionPattern": {
        "kAllReduce": 0,
        "kARResidualRMSNorm": 1,
        "kARResidualRMSNormFP8Quant": 2,
        "kARResidualRMSNormFP4Quant": 3,
        "kARResidualRMSNormOutFP8Quant": 4,
        "kARResidualRMSNormOutFP4Quant": 5,
        "kARResidualRMSNormFP8BlockWiseQuant": 6,
        "kARResidualRMSNormPartialOutFP8BlockWiseQuant": 7,
        "kARResidualRMSNormPartialOut": 8,
    },
    "AllGatherFusionPattern": {
        "kAllGather": 0,
        "kAllGatherfusedRMS": 1,
        "kAllGatherfusedRMSFP8BlockWiseQuant": 2,
    },
    "ReduceScatterFusionPattern": {
        "kReduceScatter": 0,
        "kRSResidualRMSNorm": 1,
        "kRSResidualRMSNormFP8Quant": 2,
        "kRSResidualRMSNormFP4Quant": 3,
        "kRSResidualRMSNormOutFP8Quant": 4,
        "kRSResidualRMSNormOutFP4Quant": 5,
        "kRSResidualRMSNormFP8BlockWiseQuant": 6,
        "kRSAddResidualRMSNormFP8BlockWiseQuant": 7,
        "kRSAddResidualRMSNorm": 8,
    },
}


def test_pattern_enums():
    import tokenspeed_kernel.thirdparty.cuda.trtllm as tk_comm

    for cls_name, members in EXPECTED_ENUMS.items():
        tk_cls = getattr(tk_comm, cls_name)
        for name, expected_val in members.items():
            actual_val = getattr(tk_cls, name)
            assert (
                actual_val == expected_val
            ), f"{cls_name}.{name}: expected {expected_val}, got {actual_val}"

    print("  pattern_enums: PASS (all enum values match)")


# ─── Test: Allreduce ───


def _worker_allreduce(rank, world_size, port, results):
    device = setup_distributed(rank, world_size, port)
    try:
        import tokenspeed_kernel.thirdparty.cuda.trtllm as tk_comm

        hidden_dim = 4096
        max_token_num = 128
        eps = 1e-6
        dtype = torch.bfloat16

        ipc_handles, workspace_tensor = (
            tk_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world_size, max_token_num, hidden_dim, group=dist.group.WORLD
            )
        )

        all_ok = True
        for token_num in [1, 8, 16, 64]:
            torch.manual_seed(42 + rank)
            ar_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
            res_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
            gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

            # Torch reference
            ref_ar = ar_in.clone()
            dist.all_reduce(ref_ar, op=dist.ReduceOp.SUM)
            ref_res = ref_ar + res_in
            var = ref_res.float().pow(2).mean(-1, keepdim=True)
            ref_norm = (ref_res.float() * torch.rsqrt(var + eps) * gamma.float()).to(
                dtype
            )

            # tokenspeed_kernel
            tk_res = torch.empty_like(res_in)
            tk_norm = torch.empty_like(ar_in)

            dist.barrier()
            tk_comm.trtllm_allreduce_fusion(
                allreduce_in=ar_in,
                world_size=world_size,
                world_rank=rank,
                token_num=token_num,
                hidden_dim=hidden_dim,
                workspace_ptrs=workspace_tensor,
                launch_with_pdl=True,
                trigger_completion_at_end=False,
                fp32_acc=False,
                pattern_code=tk_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                use_oneshot=True,
                allreduce_out=None,
                residual_in=res_in,
                residual_out=tk_res,
                norm_out=tk_norm,
                quant_out=None,
                scale_out=None,
                rms_gamma=gamma,
                rms_eps=eps,
                scale_factor=None,
                layout_code=None,
            )
            torch.cuda.synchronize()

            # Relaxed tolerance: IPC allreduce sums in different order than NCCL,
            # so bf16 results differ slightly due to non-associativity.
            res_close = torch.allclose(tk_res, ref_res, atol=0.125, rtol=0.05)
            norm_close = torch.allclose(tk_norm, ref_norm, atol=0.125, rtol=0.05)

            if rank == 0:
                status = "PASS" if (res_close and norm_close) else "FAIL"
                max_diff_r = (tk_res - ref_res).abs().max().item()
                max_diff_n = (tk_norm - ref_norm).abs().max().item()
                print(
                    f"  allreduce tok={token_num}: {status}"
                    f" (res_maxdiff={max_diff_r:.4f}, norm_maxdiff={max_diff_n:.4f})"
                )
                if not (res_close and norm_close):
                    all_ok = False

            dist.barrier()

        tk_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
            ipc_handles, group=dist.group.WORLD
        )
        if rank == 0:
            results["allreduce"] = all_ok
    except Exception as e:
        if rank == 0:
            import traceback

            traceback.print_exc()
            results["allreduce"] = False
    finally:
        cleanup()


# ─── Test: Reduce-scatter ───


def _worker_reducescatter(rank, world_size, port, results):
    device = setup_distributed(rank, world_size, port)
    try:
        import tokenspeed_kernel.thirdparty.cuda.trtllm as tk_comm

        hidden_dim = 4096
        max_token_num = 128
        eps = 1e-6
        dtype = torch.bfloat16

        ipc_handles, workspace_tensor = (
            tk_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world_size, max_token_num, hidden_dim, group=dist.group.WORLD
            )
        )

        all_ok = True
        for token_num in [8, 16, 64]:
            tokens_per_rank = token_num // world_size
            remaining = token_num % world_size
            token_count = tokens_per_rank + (1 if rank < remaining else 0)

            torch.manual_seed(42 + rank)
            rs_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
            res_in = torch.randn(token_count, hidden_dim, dtype=dtype, device=device)
            gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

            # Torch reference: reduce-scatter via all_reduce + slice
            ref_sum = rs_in.clone()
            dist.all_reduce(ref_sum, op=dist.ReduceOp.SUM)
            # Compute the token offset for this rank's slice
            offset = sum(
                tokens_per_rank + (1 if r < remaining else 0) for r in range(rank)
            )
            ref_slice = ref_sum[offset : offset + token_count]
            ref_res = ref_slice + res_in
            var = ref_res.float().pow(2).mean(-1, keepdim=True)
            ref_norm = (ref_res.float() * torch.rsqrt(var + eps) * gamma.float()).to(
                dtype
            )

            # tokenspeed_kernel
            tk_res = torch.empty_like(res_in)
            tk_norm = torch.empty_like(res_in)

            dist.barrier()
            tk_comm.trtllm_reducescatter_fusion(
                reducescatter_in=rs_in,
                world_size=world_size,
                world_rank=rank,
                token_num=token_num,
                hidden_dim=hidden_dim,
                workspace_ptrs=workspace_tensor,
                launch_with_pdl=True,
                trigger_completion_at_end=False,
                fp32_acc=False,
                num_token_current_rank=token_count,
                pattern_code=tk_comm.ReduceScatterFusionPattern.kRSResidualRMSNorm,
                use_oneshot=True,
                reducescatter_out=None,
                add_in=None,
                residual_in=res_in,
                residual_out=tk_res,
                norm_out=tk_norm,
                quant_out=None,
                scale_out=None,
                rms_gamma=gamma,
                rms_eps=eps,
                scale_factor=None,
                layout_code=None,
            )
            torch.cuda.synchronize()

            # Relaxed tolerance: IPC reduce-scatter sums in different order than NCCL
            res_close = torch.allclose(tk_res, ref_res, atol=0.125, rtol=0.05)
            norm_close = torch.allclose(tk_norm, ref_norm, atol=0.125, rtol=0.05)

            if rank == 0:
                status = "PASS" if (res_close and norm_close) else "FAIL"
                max_diff_r = (tk_res - ref_res).abs().max().item()
                max_diff_n = (tk_norm - ref_norm).abs().max().item()
                print(
                    f"  reducescatter tok={token_num}: {status}"
                    f" (res_maxdiff={max_diff_r:.4f}, norm_maxdiff={max_diff_n:.4f})"
                )
                if not (res_close and norm_close):
                    all_ok = False

            dist.barrier()

        tk_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
            ipc_handles, group=dist.group.WORLD
        )
        if rank == 0:
            results["reducescatter"] = all_ok
    except Exception as e:
        if rank == 0:
            import traceback

            traceback.print_exc()
            results["reducescatter"] = False
    finally:
        cleanup()


# ─── Pytest-compatible multi-GPU test runner ───


def _spawn_test(worker_fn, result_key):
    """Spawn worker_fn across GPUs via mp.spawn and assert success."""
    world_size = min(_num_gpus, 4)
    port = find_free_port()
    manager = mp.Manager()
    results = manager.dict()
    torch_mp.spawn(
        worker_fn, args=(world_size, port, results), nprocs=world_size, join=True
    )
    assert results.get(result_key, False), f"{result_key} failed"


@requires_multi_gpu
def test_allreduce():
    _spawn_test(_worker_allreduce, "allreduce")


@requires_multi_gpu
def test_reducescatter():
    _spawn_test(_worker_reducescatter, "reducescatter")
