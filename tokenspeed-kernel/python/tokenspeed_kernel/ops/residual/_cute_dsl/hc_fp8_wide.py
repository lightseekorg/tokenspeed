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

"""Register native single-launch HC mixing with prequantized FP8 operands."""

from __future__ import annotations

import threading

import cutlass
import torch
from cuda.bindings import driver
from cuda.bindings.driver import CUstream
from cutlass.cute import experimental as cute_ext
from tokenspeed_kernel.ops.residual.cute_dsl_fp8_fused import (
    Plan,
    _allocate_workspace,
    _checked,
    _compiled_function,
    _mixer_operands,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.wide import (
    Fp8FusedGatedResidualKernel,
)

_PLANS = {}
_WORKSPACES = {}
_LOCK = threading.RLock()
# Phase ordinals must never reuse scratch initialized with round generations.
_EPOCH_PROTOCOL_VERSION = 2

# Token bucket to (projection tile, split count, K tile, stages).
_DOWN_TACTICS = {
    16: (64, 8, 512, 3),
    32: (64, 8, 512, 3),
    64: (64, 8, 256, 6),
    128: (64, 8, 256, 4),
    256: (128, 16, 128, 4),
}


def _query_actual_resources(
    compiled, operands, function, threads, device, tma_l2_ids, weight_priority_ids
):
    """Query occupancy from the exact captured cooperative launch allocation."""
    with torch.cuda.device(device):
        probe_stream = _checked(
            driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
        )
        graph = None
        try:
            _checked(
                driver.cuStreamBeginCapture(
                    probe_stream,
                    driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            )
            try:
                compiled(
                    *operands,
                    cutlass.Int32(1),
                    *tma_l2_ids,
                    *weight_priority_ids,
                    probe_stream,
                )
            finally:
                graph = _checked(driver.cuStreamEndCapture(probe_stream))
            _, count = _checked(driver.cuGraphGetNodes(graph, 0))
            nodes, _ = _checked(driver.cuGraphGetNodes(graph, count))
            kernels = [
                node
                for node in nodes
                if _checked(driver.cuGraphNodeGetType(node))
                == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
            ]
            if len(kernels) != 1:
                raise RuntimeError("Expected one native cooperative compute launch")
            params = _checked(driver.cuGraphKernelNodeGetParams(kernels[0]))
            if int(params.func) != int(function) or (
                params.blockDimX,
                params.blockDimY,
                params.blockDimZ,
            ) != (threads, 1, 1):
                raise RuntimeError(
                    "Captured native function or threads differ from the plan"
                )
            cooperative = _checked(
                driver.cuGraphKernelNodeGetAttribute(
                    kernels[0],
                    driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                )
            )
            if cooperative.cooperative != 1:
                raise RuntimeError(
                    "Native phase publication requires cooperative residency"
                )
            blocks = _checked(
                driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    params.func, params.blockDimX, params.sharedMemBytes
                )
            )
            return params.sharedMemBytes, blocks
        finally:
            if graph is not None:
                _checked(driver.cuGraphDestroy(graph))
            _checked(driver.cuStreamDestroy(probe_stream))


def _execute(
    x,
    xs,
    down,
    ds,
    up,
    us,
    *,
    g,
    h,
    r,
    p,
    output_dtype,
    projection_scale,
    weights_independent,
    token_bucket,
    packed_up_weight,
):
    has_injection = p == r + g
    up_weight_is_packed = (
        packed_up_weight is not None and has_injection and token_bucket <= 128
    )
    if up_weight_is_packed:
        up = packed_up_weight
    rows = x.shape[0]
    out = torch.empty((rows, h), device=x.device, dtype=output_dtype)
    inject = (
        torch.empty((rows, g), device=x.device, dtype=output_dtype) if p > r else None
    )
    if rows == 0:
        return out, inject
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    first_bucket = token_bucket == 16
    entry_barriers = token_bucket == 128
    two_token_parts = token_bucket == 256
    small_injection = has_injection and first_bucket
    release_after_reads = has_injection and entry_barriers

    down_tactic = _DOWN_TACTICS[token_bucket]
    down_issue = "raw_unrolled" if token_bucket == 64 else "cute"
    up_issue = "cute_unrolled" if two_token_parts else down_issue
    issue_style = (down_issue, up_issue)
    down_mma_unroll = 4 if down_issue == "cute" else down_tactic[2] // 32
    down_arrival_actor = "mma" if token_bucket in (32, 64, 128) else "thread0"
    partial_store_packet = 4 if two_token_parts else 1
    native_down_partials = entry_barriers
    entry_down_prefill = True
    early_entry_weights = int(token_bucket in (16, 64, 128))

    activation_wait = (
        "grid" if first_bucket else "warp_sync" if token_bucket == 32 else "warp_arrive"
    )
    async_phase_arrivals = (
        "both" if two_token_parts else "activation" if entry_barriers else "native"
    )
    quant_max_mode = "warp" if first_bucket else "warp_local"
    quant_scale_mode = "original" if first_bucket else "scales_clamp"
    quant_max_join = "replicate" if entry_barriers else "native"
    quant_publication = "native"
    up_wait_once = entry_barriers
    entry_scale_prefetches = (entry_barriers,) * 3

    channel_parts = 2 if token_bucket <= 64 else 1
    stages = 5 if first_bucket else min(3, r // 128)
    up_tactic = (channel_parts, 128, stages)
    up_prefetch_stages = stages if first_bucket else 3 if release_after_reads else 2
    up_prefetch_split = (1, 1, 0) if small_injection else (0, 1, -1)
    up_activation_prefetch = (
        (5, 2, 1) if first_bucket else (7, 1, 1) if two_token_parts else (3, 1, 1)
    )
    up_followup_prefetch = {
        16: ("off", -1, 1),
        32: ("reader", 5, 1),
        64: ("before_acquire", 5, 1),
        128: ("before_acquire", 5, 1),
        256: ("down_handoff", 6, 1),
    }[token_bucket]
    if up_weight_is_packed and token_bucket == 32:
        last_up_weight_prefetch = "after_release", 16
    elif release_after_reads:
        last_up_weight_prefetch = "before_invalidate", 16
    else:
        last_up_weight_prefetch = "off", 0
    up_tmem_dealloc_site = "after_reads" if release_after_reads else "kernel_tail"
    up_unused_empty = int(release_after_reads)
    up_residual_order = "before_acquire" if release_after_reads else "native"
    up_hidden_mix = "native"
    if small_injection:
        up_hidden_mix = "register_half_group_no_join"
    elif has_injection and token_bucket in (32, 64):
        up_hidden_mix = "vector_pair"
    up_final_part_mode = "split" if small_injection else "native"
    up_mix_input_prefetch = "both" if small_injection else "none"
    up_reader_pair_arithmetic = "scalar_pair" if small_injection else "separate"
    up_hidden_pair_mul = has_injection and token_bucket == 64
    up_refill_unroll = two_token_parts
    up_tile4_hint_stride = 2 if two_token_parts else 0
    up_skip_final = not entry_barriers or release_after_reads
    up_weight_handoff = (
        "retire_first_stage"
        if token_bucket <= 32
        else "retire_part0" if token_bucket == 64 else "reuse_pair"
    )
    up_w2_issue_site = "handoff" if release_after_reads else "activation_demand"
    up_barrier_placement = "entry" if entry_barriers else "handoff"
    up_scale_cache = (
        "vector_register_rows"
        if two_token_parts
        else "vector" if entry_barriers else "native"
    )
    private_epoch_order = "store"
    release_phase_arrivals = True
    operand_transport = ("explicit", "explicit")
    minimum_blocks_per_sm = 1
    tma_l2_ids = (cutlass.Int32(0),) * 4
    weight_cache_hint = True
    weight_priority_ids = (cutlass.Int32(1),) * 2
    kernel_options = dict(
        g=g,
        h=h,
        r=r,
        p=p,
        token_bucket=token_bucket,
        projection_scale=projection_scale,
        use_pdl=pdl_enabled(),
        weights_independent=weights_independent,
        down_per_channel=ds.numel() > 1,
        up_per_channel=us.numel() > 1,
        down_tactic=down_tactic,
        down_mma_unroll=down_mma_unroll,
        up_tactic=up_tactic,
        up_weight_is_packed=up_weight_is_packed,
        up_prefetch_stages=up_prefetch_stages,
        up_prefetch_split=up_prefetch_split,
        up_activation_prefetch=up_activation_prefetch,
        up_followup_prefetch=up_followup_prefetch,
        last_up_weight_prefetch=last_up_weight_prefetch,
        up_tmem_dealloc_site=up_tmem_dealloc_site,
        up_unused_empty=up_unused_empty,
        up_residual_order=up_residual_order,
        up_hidden_mix=up_hidden_mix,
        up_final_part_mode=up_final_part_mode,
        up_mix_input_prefetch=up_mix_input_prefetch,
        up_reader_pair_arithmetic=up_reader_pair_arithmetic,
        up_hidden_pair_mul=up_hidden_pair_mul,
        up_w2_issue_site=up_w2_issue_site,
        up_refill_unroll=up_refill_unroll,
        up_tile4_hint_stride=up_tile4_hint_stride,
        up_skip_final=up_skip_final,
        up_weight_handoff=up_weight_handoff,
        up_barrier_placement=up_barrier_placement,
        down_arrival_actor=down_arrival_actor,
        issue_style=issue_style,
        partial_store_packet=partial_store_packet,
        native_down_partials=native_down_partials,
        up_scale_cache=up_scale_cache,
        private_epoch_order=private_epoch_order,
        release_phase_arrivals=release_phase_arrivals,
        async_phase_arrivals=async_phase_arrivals,
        operand_transport=operand_transport,
        minimum_blocks_per_sm=minimum_blocks_per_sm,
        weight_cache_hint=weight_cache_hint,
        entry_down_prefill=entry_down_prefill,
        early_entry_weights=early_entry_weights,
        activation_wait=activation_wait,
        quant_scale_mode=quant_scale_mode,
        quant_max_mode=quant_max_mode,
        quant_max_join=quant_max_join,
        quant_publication=quant_publication,
        up_wait_once=up_wait_once,
        entry_scale_prefetches=entry_scale_prefetches,
    )
    key = (
        x.device.index,
        output_dtype,
        _EPOCH_PROTOCOL_VERSION,
        *kernel_options.values(),
    )
    stream_key = key + (int(stream),)
    with _LOCK:
        plan = _PLANS.get(key)
        workspace = _WORKSPACES.get(stream_key)
        if plan is None or workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm the prequantized FP8 bucket on this stream before capture"
                )
            kernel = Fp8FusedGatedResidualKernel(**kernel_options)
            if plan is None:
                # Compile with one safe cooperative block, then query real capacity.
                scratch = _allocate_workspace(
                    token_bucket, r, p, kernel.splits, 2, x.device
                )
                operands = _mixer_operands(
                    x, xs, down, ds, up, us, scratch, out, inject
                )
                compiled = cute_ext.compile(
                    kernel,
                    *operands,
                    cutlass.Int32(1),
                    *tma_l2_ids,
                    *weight_priority_ids,
                    stream,
                )
                function = _compiled_function(compiled, x.device)
                actual_shared, blocks = _query_actual_resources(
                    compiled,
                    operands,
                    function,
                    kernel.threads,
                    x.device,
                    tma_l2_ids,
                    weight_priority_ids,
                )
                properties = torch.cuda.get_device_properties(x.device)
                capacity = blocks * properties.multi_processor_count
                if capacity <= 0:
                    raise RuntimeError(
                        "prequantized FP8 cooperative kernel has no resident capacity"
                    )
                grid = min(capacity, max(kernel.down_tasks, kernel.up_tasks))
                resources = {
                    "token_bucket": token_bucket,
                    "prepared_up_weight": up_weight_is_packed,
                    "registers": _checked(
                        driver.cuFuncGetAttribute(
                            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            function,
                        )
                    ),
                    "local_bytes": _checked(
                        driver.cuFuncGetAttribute(
                            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            function,
                        )
                    ),
                    "query_shared_bytes": actual_shared,
                    "blocks_per_sm": blocks,
                    "sm_count": properties.multi_processor_count,
                    "grid": grid,
                    "threads": kernel.threads,
                }
                plan = Plan(compiled, grid, kernel, resources)
                _PLANS[key] = plan
            with torch.inference_mode(False):
                workspace = _allocate_workspace(
                    token_bucket, r, p, plan.kernel.splits, plan.grid + 1, x.device
                )
            _WORKSPACES[stream_key] = workspace
    operands = _mixer_operands(x, xs, down, ds, up, us, workspace, out, inject)
    plan.compiled(
        *operands, cutlass.Int32(plan.grid), *tma_l2_ids, *weight_priority_ids, stream
    )
    return out, inject
