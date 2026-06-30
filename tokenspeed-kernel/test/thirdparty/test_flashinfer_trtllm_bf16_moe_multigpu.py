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

"""Multi-GPU reproducer for the qwen3.5-397b-a17b-nvfp4 agentic perf IMA.

The single-GPU sibling
`test_flashinfer_trtllm_bf16_moe.py::test_trtllm_bf16_moe_cuda_graph`
captures `flashinfer.trtllm_bf16_moe` alone in a CUDAGraph and passes
cleanly on B200, while the 8-GPU agentic perf job blows up the first
time `CudaGraphWrapper` captures the MTP/NextN drafter at bs=1:

    RuntimeError: Error in function 'run' at trtllm_batched_gemm_runner.cu:278:
    Error occurred when running GEMM! (numBatches: 512 , GemmMNK: 1 4096 128 ,
    Kernel: bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_s6_et128x8_m128x8x16
    _c1x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_rgTma_clmp_dynB_sm100f)

The only meaningful difference between the two paths is that the runtime
captures `trtllm_allreduce_fusion` (pre-MoE) and
`trtllm_reducescatter_fusion` (post-MoE) into the same graph as MoE, and
does so once per MTP step (3 steps per draft). This UT spawns
`world_size` GPUs, creates the same IPC + lamport workspaces, then
captures `[allreduce_fusion -> trtllm_bf16_moe -> reducescatter_fusion]`
repeated `spec_num_steps` times in a single CUDAGraph and replays it.

If the IMA is caused by IPC/lamport buffers overlapping or racing with
MoE workspace, this UT will reproduce it on the same b200-8gpu runner
without spinning up the full TokenSpeed server.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not platform.is_blackwell,
    reason="flashinfer trtllm_bf16_moe is only registered for SM100 Blackwell.",
)

_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


# Qwen3.5-397B-A17B-NVFP4 MoE shape under attn-tp=moe-tp=8 (and bs=1 for
# the MTP drafter's first capture).
NUM_EXPERTS = 512
TOP_K = 10
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE_PER_RANK = 128  # moe_intermediate_size (1024) // tp_size (8)
BLOCK_K = 128
SPEC_NUM_STEPS = 3
DRAFT_BS = 1
MAX_TOKEN_NUM = 2048  # mirrors the runtime's `max_token_num` for fusion workspace.


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _build_inputs(
    *,
    bs: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Build the call-site arguments for `flashinfer.trtllm_bf16_moe`.

    Weight layout follows `WeightLayout.BlockMajorK`, which the launcher
    reads as `[E, K / block_k, Mn, block_k]`.
    """
    hidden_states = torch.randn(bs, HIDDEN_SIZE, device=device, dtype=dtype)
    routing_logits = torch.randn(bs, NUM_EXPERTS, device=device, dtype=dtype)
    gemm1_weights = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE // BLOCK_K,
        2 * INTERMEDIATE_SIZE_PER_RANK,
        BLOCK_K,
        device=device,
        dtype=dtype,
    )
    gemm2_weights = torch.randn(
        NUM_EXPERTS,
        INTERMEDIATE_SIZE_PER_RANK // BLOCK_K,
        HIDDEN_SIZE,
        BLOCK_K,
        device=device,
        dtype=dtype,
    )
    gemm1_weights.mul_(1.0 / (HIDDEN_SIZE**0.5))
    gemm2_weights.mul_(1.0 / (INTERMEDIATE_SIZE_PER_RANK**0.5))
    return dict(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE_SIZE_PER_RANK,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        # 1 = Renormalize (TopK -> Softmax), what the runtime picks.
        routing_method_type=1,
    )


_VALID_MODES = ("warmup", "no_warmup")


def _worker_capture_fusion_with_moe(
    rank: int, world_size: int, port: int, results
) -> None:
    # Mode is read from the environment so multiple pytest cases can share
    # the same worker without re-spawning, while still selecting between
    # the warmup / no-warmup variants.
    mode = os.environ.get("MULTIGPU_UT_MODE", "warmup")
    assert mode in _VALID_MODES, f"unknown MULTIGPU_UT_MODE {mode!r}"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        import tokenspeed_kernel.thirdparty.cuda.trtllm as tk_comm
        from flashinfer.fused_moe.core import trtllm_bf16_moe

        # Mirror runtime: TRT-LLM AR fusion IPC workspace + lamport buffer.
        ipc_handles, workspace_tensor = (
            tk_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                MAX_TOKEN_NUM,
                HIDDEN_SIZE,
                group=dist.group.WORLD,
            )
        )

        eps = 1e-6
        dtype = torch.bfloat16

        torch.manual_seed(0xC0DE + rank)
        moe_inputs = _build_inputs(bs=DRAFT_BS, device=device, dtype=dtype)

        rms_gamma = torch.randn(HIDDEN_SIZE, device=device, dtype=dtype)
        residual = torch.randn(DRAFT_BS, HIDDEN_SIZE, device=device, dtype=dtype)

        # Buffers reused inside the captured graph (so addresses stay stable
        # under replay, just like the runtime's static input buffers).
        ar_in = moe_inputs["hidden_states"].clone()
        residual_buf = residual.clone()
        norm_out = torch.empty_like(ar_in)
        rs_out_residual = torch.empty_like(residual_buf)
        rs_out_norm = torch.empty_like(residual_buf)

        def one_step() -> None:
            # Pre-MoE all-reduce + residual + RMSNorm fusion.
            tk_comm.trtllm_allreduce_fusion(
                allreduce_in=ar_in,
                world_size=world_size,
                world_rank=rank,
                token_num=DRAFT_BS,
                hidden_dim=HIDDEN_SIZE,
                workspace_ptrs=workspace_tensor,
                launch_with_pdl=True,
                trigger_completion_at_end=False,
                fp32_acc=False,
                pattern_code=tk_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                use_oneshot=True,
                allreduce_out=None,
                residual_in=residual_buf,
                residual_out=residual_buf,
                norm_out=norm_out,
                quant_out=None,
                scale_out=None,
                rms_gamma=rms_gamma,
                rms_eps=eps,
                scale_factor=None,
                layout_code=None,
            )

            # Plug the normalized output back into MoE as hidden_states so
            # the call sequence matches the runtime exactly.
            moe_inputs["hidden_states"] = norm_out
            moe_out = trtllm_bf16_moe(
                **moe_inputs, tune_max_num_tokens=max(1, DRAFT_BS)
            )
            if isinstance(moe_out, (list, tuple)):
                moe_out = moe_out[0]

            # Reduce-scatter the expert output back, again with residual+norm
            # fused on top, again hitting the same lamport buffer.
            tk_comm.trtllm_reducescatter_fusion(
                reducescatter_in=moe_out,
                world_size=world_size,
                world_rank=rank,
                token_num=DRAFT_BS,
                hidden_dim=HIDDEN_SIZE,
                workspace_ptrs=workspace_tensor,
                launch_with_pdl=True,
                trigger_completion_at_end=False,
                fp32_acc=False,
                num_token_current_rank=DRAFT_BS,
                pattern_code=tk_comm.ReduceScatterFusionPattern.kRSResidualRMSNorm,
                use_oneshot=True,
                reducescatter_out=None,
                add_in=None,
                residual_in=rs_out_residual,
                residual_out=rs_out_residual,
                norm_out=rs_out_norm,
                quant_out=None,
                scale_out=None,
                rms_gamma=rms_gamma,
                rms_eps=eps,
                scale_factor=None,
                layout_code=None,
            )

        if mode == "warmup":
            # Warm up on a side stream, exactly as CudaGraphWrapper does.
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(SPEC_NUM_STEPS):
                    one_step()
            torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        dist.barrier()

        # Capture all SPEC_NUM_STEPS calls in a single graph, matching the
        # runtime's per-bs capture of `_run_multi_step_decode`. For
        # `no_warmup` the very first MoE call lands inside the capture
        # stream, exactly like the runtime does for MTP/NextN drafter
        # capture at bs=1.
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            for _ in range(SPEC_NUM_STEPS):
                one_step()

        # Replay a few times to surface any IMA that only fires on replay.
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        ok = (
            torch.isfinite(rs_out_residual).all().item()
            and torch.isfinite(rs_out_norm).all().item()
        )

        tk_comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
            ipc_handles, group=dist.group.WORLD
        )

        if rank == 0:
            results["ok"] = bool(ok)
    except Exception:  # pragma: no cover - surfaced via results dict
        import traceback

        traceback.print_exc()
        if rank == 0:
            results["ok"] = False
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, *, mode: str) -> None:
    assert mode in _VALID_MODES, f"unknown mode {mode!r}"
    port = _find_free_port()
    manager = mp.Manager()
    results = manager.dict()
    os.environ["MULTIGPU_UT_MODE"] = mode
    try:
        torch_mp.spawn(
            _worker_capture_fusion_with_moe,
            args=(world_size, port, results),
            nprocs=world_size,
            join=True,
        )
    finally:
        os.environ.pop("MULTIGPU_UT_MODE", None)
    assert results.get(
        "ok", False
    ), f"trtllm_bf16_moe + AR fusion graph capture failed (mode={mode})"


_skip_unless_8gpu = pytest.mark.skipif(
    _NUM_GPUS < 8,
    reason=(
        "Reproducing the qwen3.5-397b-a17b-nvfp4 MTP IMA needs the same"
        " 8-way TP as the perf job; smaller world sizes hit different"
        " AR fusion code paths."
    ),
)


def _ensure_spawn() -> None:
    # Force `spawn` so CUDA state does not leak from the test runner
    # (matches the pattern in `test_trtllm_comm.py`).
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)


@_skip_unless_8gpu
def test_trtllm_bf16_moe_cuda_graph_8gpu_with_ar_fusion() -> None:
    """Warm-up-then-capture variant: matches CudaGraphWrapper exactly."""
    _ensure_spawn()
    _spawn(world_size=8, mode="warmup")


@_skip_unless_8gpu
def test_trtllm_bf16_moe_cuda_graph_8gpu_no_warmup() -> None:
    """Cold-capture variant: drop the side-stream warmup so the first
    `trtllm_bf16_moe` call happens inside the capture stream, the same
    way the MTP/NextN drafter sees the bs=1 shape for the first time in
    `CudaGraphWrapper.capture()`."""
    _ensure_spawn()
    _spawn(world_size=8, mode="no_warmup")
