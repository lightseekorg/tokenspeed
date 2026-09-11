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

"""Real DeepEP/DeepGEMM and NCCL qualification on four SM90+ GPUs.

Run ``python3 -m pytest -q test/runtime/distributed/test_deepep_prefill_bcg.py``.
The dedicated CI job sets ``TOKENSPEED_DEEPEP_BCG_REQUIRE_4GPU=1`` so missing
resources fail instead of producing a green skip. The parent watchdog also
bounds vendor-kernel hangs that process-group timeouts cannot interrupt.

This uses a tiny Qwen sparse block with generated FP8 expert weights, real
TopK, the production MoELayer and model BCG adapter, and real shared-expert
work. Attention TP2 reduce-scatter/all-gather surround the break, while all
four ranks form one EP group and two unequal attention DP replicas. No model
download, fake dispatcher, or numerical substitute stands in for the kernels.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 4
_TP_SIZE = 2
_HIDDEN = 2048  # Smallest hidden width supported by the DeepEP LL kernels.
_INTERMEDIATE = 256
_EXPERTS = 16
_TOP_K = 2
_TIMEOUT_SECONDS = 900
_BARRIER_TIMEOUT = timedelta(seconds=120)


class _Router(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Input-selected routes change on replay without replacing the module.
        return hidden_states[:, :_EXPERTS].float().contiguous(), None


class _SharedExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # A fresh result each invocation exercises the shared-output lifetime
        # which would be stale if only MoELayer.forward were an eager break.
        return torch.sin(hidden_states) * 0.125


@dataclass
class _Bucket:
    capture: Any
    ctx: Any
    inputs: torch.Tensor
    shard: torch.Tensor
    output: torch.Tensor
    handoff: torch.Tensor
    handoff_ptr: int


def _context(bucket: int, decode: bool) -> Any:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0 if decode else 1,
        input_num_tokens=bucket,
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        global_num_tokens=[bucket] * _WORLD_SIZE,
        global_bs=[1] * _WORLD_SIZE,
        all_decode_or_idle=decode,
        all_extend=not decode,
    )


def _quantize_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.thirdparty.deep_gemm import ceil_to_ue8m0

    experts, rows, cols = weights.shape
    blocks = weights.view(experts, rows // 128, 128, cols // 128, 128).float()
    amax = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6)
    scales = ceil_to_ue8m0(amax / 448.0)
    quantized = (blocks / scales).to(torch.float8_e4m3fn).view_as(weights)
    return quantized, scales.view(experts, rows // 128, cols // 128).contiguous()


def _make_block(rank: int) -> Any:
    import tokenspeed_kernel

    from tokenspeed.runtime.distributed.comm_manager import CommManager
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.layers.moe.topk import TopK
    from tokenspeed.runtime.layers.moe.utils import initialize_moe_config
    from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    initialize_moe_config(
        SimpleNamespace(
            all2all_backend="deepep",
            moe_backend="deep_gemm",
            deepep_mode="auto",
            disable_flashinfer_cutlass_moe_fp4_allgather=False,
        )
    )
    mapping = Mapping(
        rank=rank,
        world_size=_WORLD_SIZE,
        attn_tp_size=_TP_SIZE,
        attn_cp_size=1,
        attn_dp_size=2,
        dense_tp_size=4,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        moe_dp_size=1,
    )
    # Bypass only checkpoint/config loading. Both forward methods below are
    # the production methods, with the ordinary registry-selected kernel plan.
    experts = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(experts)
    experts.top_k = _TOP_K
    experts.num_experts = _EXPERTS
    experts.num_local_experts = _EXPERTS // _WORLD_SIZE
    experts.ep_rank = rank
    experts.ep_size = _WORLD_SIZE
    experts.activation = "silu"
    generator = torch.Generator(device="cuda").manual_seed(20260910 + rank)
    for name, rows, cols in (
        ("w13_weight", 2 * _INTERMEDIATE, _HIDDEN),
        ("w2_weight", _HIDDEN, _INTERMEDIATE),
    ):
        raw = (
            torch.randn(
                experts.num_local_experts,
                rows,
                cols,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.025
        )
        quantized, scales = _quantize_weights(raw)
        setattr(experts, name, torch.nn.Parameter(quantized, requires_grad=False))
        setattr(
            experts,
            name + "_scale_inv",
            torch.nn.Parameter(scales, requires_grad=False),
        )
    experts.plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="deepep",
        ep_size=_WORLD_SIZE,
        ispp=_INTERMEDIATE,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        with_bias=False,
        deepep_group=dist.group.WORLD,
        deepep_mode="auto",
        deepep_low_latency_max_num_tokens_per_gpu=16,
        solution="deep_gemm",
    )
    assert experts.plan["apply_kernel_name"] == "deep_gemm_deepep_fp8_moe_apply"
    tokenspeed_kernel.moe_process_weights(experts.plan, experts)

    block = Qwen3_5MoeSparseMoeBlock.__new__(Qwen3_5MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.mapping = mapping
    block.use_deepep = True
    block.comm_manager = CommManager(
        mapping=mapping,
        layer_id=0,
        is_moe=True,
        prev_is_moe=True,
        input_layernorm=None,
        post_attn_layernorm=None,
    )
    block.gate = _Router()
    block.experts = experts
    block.topk = TopK(
        top_k=_TOP_K,
        renormalize=True,
        use_grouped_topk=False,
        output_format=experts.topk_output_format,
    )
    block.shared_expert = _SharedExpert()
    block.shared_expert_gate = torch.nn.Linear(
        _HIDDEN, 1, bias=False, device="cuda", dtype=torch.bfloat16
    )
    block.shared_expert_gate.weight.data.zero_()
    return block.eval()


def _inputs(rows: int, dp_rank: int, destination: int, phase: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(1000 + dp_rank * 100 + phase)
    hidden = (
        torch.randn(
            rows, _HIDDEN, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        * 0.2
    )
    hidden[:, :_EXPERTS] = -4
    first = destination * (_EXPERTS // _WORLD_SIZE)
    hidden[:, first] = 4
    hidden[:, first + 1] = 3
    # Unique row markers survive the TP communication and shared-expert path.
    hidden[:, _EXPERTS] = torch.arange(rows, device="cuda") * 0.03125 + dp_rank
    return hidden


def _observe_dispatch(dispatcher: Any) -> dict[str, Any]:
    """Observe the real vendor call, without changing its data or result."""
    record: dict[str, Any] = {"normal_calls": 0, "sources": 0, "routes_received": 0}
    original_a = dispatcher.dispatch_a
    original_b = dispatcher.dispatch_b
    in_normal = False

    def dispatch_a(hidden_states, topk_idx, topk_weights, low_latency):
        nonlocal in_normal
        in_normal = not low_latency
        if in_normal:
            record["normal_calls"] += 1
            record["sources"] = hidden_states.shape[0]
        return original_a(hidden_states, topk_idx, topk_weights, low_latency)

    def dispatch_b():
        result = original_b()
        if in_normal:
            # Structural capture must never reach this CPU read. Count actual
            # received top-k slots, rather than the 128-aligned GEMM row count.
            record["routes_received"] = int((result[1] >= 0).sum().item())
        return result

    dispatcher.dispatch_a = dispatch_a
    dispatcher.dispatch_b = dispatch_b
    return record


def _padded_forward(block, inputs, shard, output, ctx, tp_group):
    # Both TP peers supply half the same attention output; RS reconstructs
    # the complete value on each physical shard exactly (division by two).
    dist.reduce_scatter_tensor(shard, inputs, group=tp_group)
    local_output = block(
        shard,
        num_global_tokens=ctx.input_num_tokens * 2,
        max_num_tokens_per_gpu=shard.shape[0],
        ctx=ctx,
    )
    dist.all_gather_into_tensor(output, local_output.contiguous(), group=tp_group)
    return local_output


def _capture_bucket(block, bucket, rank, tp_group, cpu_group, pool, record):
    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        BreakableCapture,
        active_forward,
        weak_ref_tensor,
    )

    ctx = _context(bucket, False)
    inputs = _inputs(bucket, rank // _TP_SIZE, 1, 0) / _TP_SIZE
    shard = torch.empty(
        (bucket // _TP_SIZE, _HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty_like(inputs)
    for _ in range(3):
        _padded_forward(block, inputs, shard, output, ctx, tp_group)
    torch.cuda.synchronize()
    # Make startup timing asymmetric; peers must wait until every rank has
    # finished eager dispatch before any rank enters structural capture.
    if rank == _WORLD_SIZE - 1:
        time.sleep(0.25)
    dist.monitored_barrier(group=cpu_group, timeout=_BARRIER_TIMEOUT)
    normal_before = record["normal_calls"]
    shared_before = block.shared_expert.calls
    capture = BreakableCapture(pool=pool, stream=None)
    with active_forward(ctx), capture:
        local_output = _padded_forward(block, inputs, shard, output, ctx, tp_group)
    handoff = weak_ref_tensor(local_output)
    handoff_ptr = local_output.data_ptr()
    del local_output
    assert len(capture.segments) == 3, "expected graph / real MoE break / graph"
    assert (
        record["normal_calls"] == normal_before
    ), "DeepEP ran during structural capture"
    assert block.shared_expert.calls == shared_before, "capture stub ran shared expert"
    dist.monitored_barrier(group=cpu_group, timeout=_BARRIER_TIMEOUT)
    with active_forward(ctx):
        capture.replay(valid_rows=None)
    torch.cuda.synchronize()
    assert record["normal_calls"] == normal_before + 1
    return _Bucket(capture, ctx, inputs, shard, output, handoff, handoff_ptr)


def _capture_decode(block, rank, cpu_group):
    from tokenspeed.runtime.execution.forward_step import DeepEPCudaGraphRunnerAdapter

    ctx = _context(_TP_SIZE, True)
    inputs = _inputs(1, rank // _TP_SIZE, 1, 97)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            block(inputs, num_global_tokens=4, max_num_tokens_per_gpu=1, ctx=ctx)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.monitored_barrier(group=cpu_group, timeout=_BARRIER_TIMEOUT)
    graph = torch.cuda.CUDAGraph()
    adapter = DeepEPCudaGraphRunnerAdapter()
    adapter.capture()
    assert adapter._active, "decode adapter must bind the initialized DeepEP buffer"
    with torch.cuda.graph(graph, stream=stream):
        output = block(inputs, num_global_tokens=4, max_num_tokens_per_gpu=1, ctx=ctx)
    torch.cuda.synchronize()
    dist.monitored_barrier(group=cpu_group, timeout=_BARRIER_TIMEOUT)
    return graph, adapter, inputs, output, ctx


def _check_replay(block, entry, rank, counts, destination, phase, record):
    from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward

    bucket = entry.inputs.shape[0]
    dp_rank, tp_rank = divmod(rank, _TP_SIZE)
    real_rows = counts[dp_rank]
    full_input = _inputs(bucket, dp_rank, destination, phase)
    start = tp_rank * (bucket // _TP_SIZE)
    live = min(max(real_rows - start, 0), bucket // _TP_SIZE)
    expected_local = block(
        full_input[start : start + live],
        num_global_tokens=sum(counts),
        max_num_tokens_per_gpu=max(counts),
        ctx=entry.ctx,
    ).clone()
    entry.inputs.copy_(full_input / _TP_SIZE)
    entry.inputs[real_rows:].fill_(float("nan"))
    entry.handoff.fill_(float("nan"))
    calls_before = record["normal_calls"]
    shared_before = block.shared_expert.calls
    # Use a fresh context every replay so a closure that accidentally retains
    # the capture context cannot hide behind object mutation in this test.
    live_ctx = _context(bucket, False)
    with active_forward(live_ctx):
        entry.capture.replay(valid_rows=real_rows)
    torch.cuda.synchronize()
    assert record["normal_calls"] == calls_before + 1, "replay did not execute DeepEP"
    assert block.shared_expert.calls == shared_before + 1
    assert entry.handoff.data_ptr() == entry.handoff_ptr
    assert record["sources"] == live
    expected_routes = sum(counts) * _TOP_K if rank == destination else 0
    assert record["routes_received"] == expected_routes
    if live == 0 and rank == destination:
        assert expected_routes > 0, "empty source must still execute remote expert work"
    assert torch.count_nonzero(entry.handoff[live:]).item() == 0
    assert torch.count_nonzero(entry.output[real_rows:]).item() == 0
    assert torch.isfinite(entry.output).all().item()
    # The same FP8 kernels run in each arm. Use their existing BF16 output
    # tolerance (test_deepep_normal_compute), with exact count/tail invariants
    # above so approximate numerics cannot conceal padding or routing errors.
    torch.testing.assert_close(
        entry.handoff[:live].float(),
        expected_local.float(),
        rtol=2e-2,
        atol=2e-2 * expected_local.float().abs().max().item() if live else 0,
    )
    torch.testing.assert_close(
        entry.output[start : start + live], entry.handoff[:live], rtol=0, atol=0
    )


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    from tokenspeed_kernel.ops.communication.deep_ep import (
        DeepEPBuffer,
        DeepEPDispatchMode,
    )

    dist.init_process_group(
        backend="nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=_BARRIER_TIMEOUT,
    )
    try:
        cpu_group = dist.new_group(list(range(_WORLD_SIZE)), backend="gloo")
        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        tp_group = tp_groups[rank // _TP_SIZE]
        with torch.inference_mode():
            block = _make_block(rank)
            ctx = _context(8, False)
            # Initialize vendor buffers and kernels with a genuine normal call.
            block(_inputs(4, rank // _TP_SIZE, 1, 0), 16, 4, ctx)
            dispatcher = block.experts.plan["_deepep_dispatcher"]
            record = _observe_dispatch(dispatcher)
            pool = torch.cuda.graph_pool_handle()
            # Include a bucket not divisible by eight; each TP shard has five
            # rows. Expert alignment 128 must not be mistaken for token padding.
            buckets = {
                size: _capture_bucket(
                    block, size, rank, tp_group, cpu_group, pool, record
                )
                for size in (16, 10, 8)
            }
            decode_graph, adapter, decode_inputs, decode_output, decode_ctx = (
                _capture_decode(block, rank, cpu_group)
            )
            cases = (
                (16, (1, 5), 1),  # rank 1 sends zero rows but receives all routes
                (8, (5, 7), 3),  # physical live counts [4,1], not [3,2]
                (10, (9, 3), 1),
                (16, (15, 16), 3),
                (8, (3, 1), 1),
            )
            for phase in range(100):
                size, counts, destination = cases[phase % len(cases)]
                _check_replay(
                    block, buckets[size], rank, counts, destination, phase, record
                )

                # Normal BCG -> real low-latency whole-step decode graph. The
                # existing adapter must clean the shared buffer before replay.
                decode_inputs.copy_(_inputs(1, rank // _TP_SIZE, destination, phase))
                assert DeepEPBuffer._dispatch_mode == DeepEPDispatchMode.NORMAL
                adapter.replay()
                assert DeepEPBuffer._dispatch_mode == DeepEPDispatchMode.LOW_LATENCY
                decode_graph.replay()
                actual_decode = decode_output.clone()
                torch.cuda.synchronize()
                # Compute the eager LL oracle only AFTER graph replay: running
                # it first would clean the normal buffer itself and conceal a
                # broken adapter transition.
                eager_decode = block(
                    decode_inputs,
                    num_global_tokens=4,
                    max_num_tokens_per_gpu=1,
                    ctx=decode_ctx,
                )
                torch.testing.assert_close(
                    actual_decode.float(), eager_decode.float(), rtol=2e-2, atol=2e-2
                )

                # An over-bucket extend follows the existing eager method and
                # switches back to normal; the next loop re-enters BCG.
                eager_ctx = _context(34, False)
                eager_output = block(
                    _inputs(17, rank // _TP_SIZE, destination, phase + 1),
                    num_global_tokens=68,
                    max_num_tokens_per_gpu=17,
                    ctx=eager_ctx,
                )
                assert torch.isfinite(eager_output).all().item()
                assert not hasattr(dispatcher, "_dispatch_intermediate_state")
                assert not hasattr(dispatcher, "_combine_intermediate_state")
                assert dispatcher._normal_dispatcher.handle is None
                assert dispatcher._low_latency_dispatcher.handle is None
            torch.cuda.synchronize()
            dist.monitored_barrier(group=cpu_group, timeout=_BARRIER_TIMEOUT)
    finally:
        dist.destroy_process_group()


def test_deepep_prefill_bcg_tp2_dp2_ep4(tmp_path: Path) -> None:
    required = os.environ.get("TOKENSPEED_DEEPEP_BCG_REQUIRE_4GPU") == "1"
    if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
        message = "DeepEP BCG qualification requires four CUDA GPUs on one NVLink node"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    if any(
        torch.cuda.get_device_capability(index)[0] < 9 for index in range(_WORLD_SIZE)
    ):
        pytest.fail("DeepEP FP8 BCG qualification requires four SM90+ GPUs")
    # Missing DeepEP/DeepGEMM or a registry mismatch fails in the children. Do
    # not importorskip: a dedicated CUDA CI job must exercise the real kernels.
    rendezvous = (tmp_path / "deepep-bcg-rendezvous").as_uri()
    processes = mp.spawn(_worker, args=(rendezvous,), nprocs=_WORLD_SIZE, join=False)
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    try:
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(
                    f"DeepEP BCG workers exceeded {_TIMEOUT_SECONDS}s; possible collective hang"
                )
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
