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
    Workspace,
    _checked,
    _view,
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


def _select_entry_down_prefill(g, h, r, token_bucket):
    """Overlap first Down operand TMA with entry TMEM/epoch setup."""
    return (g, h, r) == (4, 8192, 1024)


def _select_early_entry_weights(g, h, r, token_bucket):
    return int((g, h, r) == (4, 8192, 1024) and token_bucket in (16, 64, 128))


def _select_activation_wait(g, h, r, token_bucket):
    if (g, h, r) == (4, 8192, 1024):
        if token_bucket == 32:
            return "warp_sync"
        if token_bucket in (64, 128, 256):
            return "warp_arrive"
    return "grid"


def _select_entry_scale_prefetches(g, h, r, token_bucket):
    selected = (g, h, r, token_bucket) == (4, 8192, 1024, 128)
    return selected, selected, selected


def _select_quant_handoff(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 8192, 1024, 128):
        return "replicate", "native", True
    return "native", "native", False


def _select_quant_scale_mode(g, h, r, token_bucket):
    if (g, h, r) == (4, 8192, 1024):
        if token_bucket in (32, 64, 256):
            return "scales_clamp"
        if token_bucket == 128:
            return "scales_clamp"
    return "original"


def _select_quant_max_mode(g, h, r, token_bucket, device):
    if (g, h, r) == (4, 8192, 1024) and torch.cuda.get_device_capability(device)[
        0
    ] == 10:
        if token_bucket == 16:
            return "warp"
        if token_bucket in (32, 64, 128, 256):
            return "warp_local"
    return "shuffle"


def _select_down_tactic(g, h, r, token_bucket):
    """Choose explicit startup-static (projection rows, splits, K, stages)."""
    if r == 1024:
        if token_bucket <= 32:
            return 64, 8, 512, 3
        if token_bucket == 256:
            return 128, 16, 128, 4
        if token_bucket == 64:
            return 64, 8, 256, 6
        return 64, 8, 256, 4
    return 64, 16, 128, 3


def _select_down_mma_unroll(g, h, r, token_bucket, issue_style):
    if issue_style == "cute":
        return 4
    # Native dense E4M3 instructions consume32K elements per issue.
    return _select_down_tactic(g, h, r, token_bucket)[2] // 32


def _select_up_tactic(g, h, r, token_bucket):
    channel_parts = 2 if g == 4 and r == 1024 and token_bucket <= 64 else 1
    k = 128 if r == 1024 else (64 if token_bucket == 128 else 320)
    stages = 5 if channel_parts == 2 and token_bucket == 16 else min(3, r // k)
    return channel_parts, k, stages


def _select_issue_style(g, h, r, token_bucket):
    if _select_up_refill_unroll(g, h, r, token_bucket):
        return "cute", "cute_unrolled"
    return (
        ("raw_unrolled", "raw_unrolled")
        if r == 1024 and token_bucket == 64
        else ("cute", "cute")
    )


def _select_down_arrival_actor(g, h, r, token_bucket):
    return (
        "mma"
        if (g, h, r) == (4, 8192, 1024) and token_bucket in (32, 64, 128)
        else "thread0"
    )


def _select_up_barrier_placement(g, h, r, token_bucket):
    return "entry" if (g, h, r, token_bucket) == (4, 8192, 1024, 128) else "handoff"


def _select_up_weight_handoff(g, h, r, token_bucket):
    if (g, h, r) != (4, 8192, 1024):
        return "native"
    return {
        16: "retire_first_stage",
        32: "retire_first_stage",
        64: "retire_part0",
        128: "reuse_pair",
        256: "reuse_pair",
    }[token_bucket]


def _select_up_prefetch_stages(g, h, r, token_bucket):
    if r == 1024:
        return _select_up_tactic(g, h, r, token_bucket)[2] if token_bucket == 16 else 2
    return 0


def _select_up_activation_prefetch(g, h, r, token_bucket):
    if (g, h, r) != (4, 8192, 1024):
        return 0, 0, 1
    return {
        16: (5, 2, 1),
        32: (3, 1, 1),
        64: (3, 1, 1),
        128: (3, 1, 1),
        256: (7, 1, 1),
    }[token_bucket]


def _select_up_prefetch_split(g, h, r, p, token_bucket, device):
    if (g, h, r, p, token_bucket) == (4, 8192, 1024, 1028, 16):
        if torch.cuda.get_device_capability(device) == (10, 3):
            return 1, 1, 0
    return 0, 1, -1


def _select_up_followup_prefetch(g, h, r, token_bucket):
    if (g, h, r) != (4, 8192, 1024):
        return "off", -1, 1
    return {
        16: ("off", -1, 1),
        32: ("reader", 5, 1),
        64: ("before_acquire", 5, 1),
        128: ("before_acquire", 5, 1),
        256: ("down_handoff", 6, 1),
    }[token_bucket]


def _select_last_up_weight_prefetch(g, h, r, p, token_bucket, up_weight_is_packed):
    if up_weight_is_packed and (g, h, r, p, token_bucket) == (4, 8192, 1024, 1028, 32):
        return "after_release", 16
    if (g, h, r, token_bucket) == (4, 8192, 1024, 128) and p == r + g:
        return "before_invalidate", 16
    return "off", 0


def _select_up_refill_unroll(g, h, r, token_bucket):
    return (g, h, r, token_bucket) == (4, 8192, 1024, 256)


def _select_up_tile4_hint_stride(g, h, r, token_bucket):
    return 2 if _select_up_refill_unroll(g, h, r, token_bucket) else 0


def _select_up_skip_final(g, h, r, token_bucket):
    return (g, h, r) == (4, 8192, 1024) and token_bucket in (16, 32, 64, 256)


def _select_up_tmem_dealloc_site(g, h, r, p, token_bucket):
    if (g, h, r, token_bucket) == (4, 8192, 1024, 128) and p == r + g:
        return "after_reads"
    return "kernel_tail"


def _select_up_w2_issue_site(g, h, r, p, token_bucket):
    if (g, h, r, p, token_bucket) == (4, 8192, 1024, 1028, 128):
        return "handoff"
    return "activation_demand"


def _select_up_unused_empty(g, h, r, p, token_bucket):
    return int(_select_up_tmem_dealloc_site(g, h, r, p, token_bucket) == "after_reads")


def _select_up_residual_order(g, h, r, p, token_bucket):
    if (g, h, r, p, token_bucket) == (4, 8192, 1024, 1028, 128):
        return "before_acquire"
    return "native"


def _select_up_hidden_mix(g, h, r, p, token_bucket):
    if (g, h, r, p) == (4, 8192, 1024, 1028) and token_bucket in (32, 64):
        return "vector_pair"
    return "native"


def _select_up_final_part_mode(g, h, r, p, token_bucket, device):
    if (g, h, r, p, token_bucket) == (
        4,
        8192,
        1024,
        1028,
        16,
    ) and torch.cuda.get_device_capability(device) == (10, 3):
        return "split"
    return "native"


def _select_up_mix_input_prefetch(g, h, r, p, token_bucket, device):
    if (g, h, r, p, token_bucket) == (
        4,
        8192,
        1024,
        1028,
        16,
    ) and torch.cuda.get_device_capability(device) == (10, 3):
        return "both"
    return "none"


def _select_up_reader_pair_arithmetic(g, h, r, p, token_bucket):
    return "separate"


def _select_up_hidden_pair_mul(device, g, h, r, p, token_bucket):
    return (g, h, r, p, token_bucket) == (4, 8192, 1024, 1028, 64) and (
        torch.cuda.get_device_capability(device) == (10, 3)
    )


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


def _select_up_scale_cache(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 8192, 1024, 256):
        return "vector_register_rows"
    return "vector" if r == 1024 and token_bucket >= 128 else "native"


def _select_partial_store_packet(g, h, r, token_bucket):
    return 4 if r == 1024 and token_bucket == 256 else 1


def _select_native_down_partials(g, h, r, token_bucket):
    return (g, h, r, token_bucket) == (4, 8192, 1024, 128)


def _select_private_epoch_order(g, h, r, token_bucket):
    return "store" if r == 1024 else "ordered"


def _select_async_phase_arrivals(device, g, h, r, token_bucket):
    if (g, h, r) == (4, 8192, 1024) and token_bucket in (128, 256):
        if torch.cuda.get_device_capability(device)[0] == 10:
            return "both" if token_bucket == 256 else "activation"
    return "native"


def _select_release_phase_arrivals(g, h, r, token_bucket):
    return r == 1024


def _select_tma_fetch(g, h, r, token_bucket):
    return (
        (("explicit", "explicit"), (0, 0, 0, 0), 1)
        if g == 4 and r == 1024
        else (("native", "native"), (0, 0, 0, 0), 0)
    )


def _select_weight_eviction(g, h, r, token_bucket):
    return (True, (1, 1)) if g == 4 and r == 1024 else (False, (0, 0))


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
    up_weight_is_packed = (
        packed_up_weight is not None
        and p == r + g
        and token_bucket in (16, 32, 64, 128)
        and torch.cuda.get_device_capability(up.device) == (10, 3)
    )
    if packed_up_weight is not None:
        if (
            (g, h, r) != (4, 8192, 1024)
            or packed_up_weight.shape != up.shape
            or packed_up_weight.dtype != up.dtype
            or packed_up_weight.device != up.device
            or not packed_up_weight.is_contiguous()
            or packed_up_weight.data_ptr() % 16
        ):
            raise ValueError("Invalid storage for packed wide E4M3 Up tiles")
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
    down_tactic = _select_down_tactic(g, h, r, token_bucket)
    issue_style = _select_issue_style(g, h, r, token_bucket)
    down_mma_unroll = _select_down_mma_unroll(g, h, r, token_bucket, issue_style[0])
    up_tactic = _select_up_tactic(g, h, r, token_bucket)
    up_prefetch_stages = _select_up_prefetch_stages(g, h, r, token_bucket)
    up_prefetch_split = _select_up_prefetch_split(g, h, r, p, token_bucket, x.device)
    up_activation_prefetch = _select_up_activation_prefetch(g, h, r, token_bucket)
    up_followup_prefetch = _select_up_followup_prefetch(g, h, r, token_bucket)
    last_up_weight_prefetch = _select_last_up_weight_prefetch(
        g, h, r, p, token_bucket, up_weight_is_packed
    )
    up_tmem_dealloc_site = _select_up_tmem_dealloc_site(g, h, r, p, token_bucket)
    up_unused_empty = _select_up_unused_empty(g, h, r, p, token_bucket)
    up_residual_order = _select_up_residual_order(g, h, r, p, token_bucket)
    up_hidden_mix = _select_up_hidden_mix(g, h, r, p, token_bucket)
    up_final_part_mode = _select_up_final_part_mode(g, h, r, p, token_bucket, x.device)
    up_mix_input_prefetch = _select_up_mix_input_prefetch(
        g, h, r, p, token_bucket, x.device
    )
    up_reader_pair_arithmetic = _select_up_reader_pair_arithmetic(
        g, h, r, p, token_bucket
    )
    if (g, h, r, p, token_bucket) == (
        4,
        8192,
        1024,
        1028,
        16,
    ) and torch.cuda.get_device_capability(x.device) == (10, 3):
        up_hidden_mix = "register_half_group_no_join"
        up_reader_pair_arithmetic = "scalar_pair"
    up_hidden_pair_mul = _select_up_hidden_pair_mul(x.device, g, h, r, p, token_bucket)
    up_refill_unroll = _select_up_refill_unroll(g, h, r, token_bucket)
    up_tile4_hint_stride = _select_up_tile4_hint_stride(g, h, r, token_bucket)
    up_skip_final = (
        _select_up_skip_final(g, h, r, token_bucket)
        or up_tmem_dealloc_site == "after_reads"
    )
    up_weight_handoff = _select_up_weight_handoff(g, h, r, token_bucket)
    up_w2_issue_site = _select_up_w2_issue_site(g, h, r, p, token_bucket)
    if up_w2_issue_site != "activation_demand":
        up_prefetch_stages = 3
    if up_w2_issue_site == "down_retire":
        up_weight_handoff = "reuse_three"
    up_barrier_placement = _select_up_barrier_placement(g, h, r, token_bucket)
    down_arrival_actor = _select_down_arrival_actor(g, h, r, token_bucket)
    if up_weight_handoff in ("prefix_three", "join_three", "reuse_three"):
        up_prefetch_stages = 3
    partial_store_packet = _select_partial_store_packet(g, h, r, token_bucket)
    native_down_partials = _select_native_down_partials(g, h, r, token_bucket)
    up_scale_cache = _select_up_scale_cache(g, h, r, token_bucket)
    private_epoch_order = _select_private_epoch_order(g, h, r, token_bucket)
    release_phase_arrivals = _select_release_phase_arrivals(g, h, r, token_bucket)
    async_phase_arrivals = _select_async_phase_arrivals(x.device, g, h, r, token_bucket)
    operand_transport, tma_l2_promotion, minimum_blocks_per_sm = _select_tma_fetch(
        g, h, r, token_bucket
    )
    if len(tma_l2_promotion) != 4 or any(
        type(v) is not int or v not in (0, 64, 128, 256) for v in tma_l2_promotion
    ):
        raise ValueError(
            "Four TMA L2 fetch policies must explicitly select0,64,128or256bytes"
        )
    tma_l2_ids = tuple(
        cutlass.Int32({0: 0, 64: 1, 128: 2, 256: 3}[v]) for v in tma_l2_promotion
    )
    weight_cache_hint, weight_priorities = _select_weight_eviction(
        g, h, r, token_bucket
    )
    if (
        type(weight_cache_hint) is not bool
        or len(weight_priorities) != 2
        or any(type(v) is not int or v not in (0, 1, 2) for v in weight_priorities)
    ):
        raise ValueError(
            "Explicit cache-hint bool and two priorities0normal/1first/2last are required"
        )
    if not weight_cache_hint and weight_priorities != (0, 0):
        raise ValueError("Disabled weight hints cannot discard non-normal priorities")
    weight_priority_ids = tuple(cutlass.Int32(v) for v in weight_priorities)
    entry_down_prefill = _select_entry_down_prefill(g, h, r, token_bucket)
    early_entry_weights = _select_early_entry_weights(g, h, r, token_bucket)
    activation_wait = _select_activation_wait(g, h, r, token_bucket)
    quant_scale_mode = _select_quant_scale_mode(g, h, r, token_bucket)
    quant_max_join, quant_publication, up_wait_once = _select_quant_handoff(
        g, h, r, token_bucket
    )
    entry_scale_prefetches = _select_entry_scale_prefetches(g, h, r, token_bucket)
    quant_max_mode = _select_quant_max_mode(g, h, r, token_bucket, x.device)
    key = (
        x.device.index,
        g,
        h,
        r,
        p,
        output_dtype,
        projection_scale,
        weights_independent,
        ds.numel() > 1,
        us.numel() > 1,
        pdl_enabled(),
        down_tactic,
        down_mma_unroll,
        up_tactic,
        up_weight_is_packed,
        up_prefetch_stages,
        up_prefetch_split,
        up_activation_prefetch,
        up_followup_prefetch,
        last_up_weight_prefetch,
        up_tmem_dealloc_site,
        up_unused_empty,
        up_residual_order,
        up_hidden_mix,
        up_final_part_mode,
        up_mix_input_prefetch,
        up_reader_pair_arithmetic,
        up_hidden_pair_mul,
        up_w2_issue_site,
        up_refill_unroll,
        up_tile4_hint_stride,
        up_skip_final,
        up_weight_handoff,
        up_barrier_placement,
        down_arrival_actor,
        issue_style,
        _EPOCH_PROTOCOL_VERSION,
        partial_store_packet,
        native_down_partials,
        up_scale_cache,
        private_epoch_order,
        release_phase_arrivals,
        async_phase_arrivals,
        operand_transport,
        minimum_blocks_per_sm,
        entry_down_prefill,
        early_entry_weights,
        activation_wait,
        quant_scale_mode,
        quant_max_mode,
        quant_max_join,
        quant_publication,
        up_wait_once,
        entry_scale_prefetches,
        weight_cache_hint,
        token_bucket,
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
            kernel = Fp8FusedGatedResidualKernel(
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
            if plan is None:
                # Compile with one safe cooperative block, then query real capacity.
                scratch = Workspace(
                    torch.empty(
                        (kernel.splits, token_bucket, p),
                        device=x.device,
                        dtype=torch.float32,
                    ),
                    torch.empty(
                        (token_bucket, r), device=x.device, dtype=torch.float8_e4m3fn
                    ),
                    torch.empty((token_bucket,), device=x.device, dtype=torch.float32),
                    torch.zeros(
                        (2,),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                )
                values = (
                    x,
                    xs.reshape(-1),
                    down,
                    ds.reshape(-1),
                    up,
                    us.reshape(-1),
                    scratch.partials,
                    scratch.activation,
                    scratch.alpha,
                    scratch.epochs,
                    out,
                    out if inject is None else inject,
                )
                operands = tuple(_view(v) for v in values)
                compiled = cute_ext.compile(
                    kernel,
                    *operands,
                    cutlass.Int32(1),
                    *tma_l2_ids,
                    *weight_priority_ids,
                    stream,
                )
                compiled.to(device=x.device.index)
                library = driver.CUlibrary(int(compiled.jit_module.cuda_library[0]))
                handle = _checked(driver.cuLibraryEnumerateKernels(1, library))[0]
                function = _checked(driver.cuKernelGetFunction(handle))
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
                workspace = Workspace(
                    torch.empty(
                        (plan.kernel.splits, token_bucket, p),
                        device=x.device,
                        dtype=torch.float32,
                    ),
                    torch.empty(
                        (token_bucket, r), device=x.device, dtype=torch.float8_e4m3fn
                    ),
                    torch.empty((token_bucket,), device=x.device, dtype=torch.float32),
                    torch.zeros(
                        (plan.grid + 1,),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                )
            _WORKSPACES[stream_key] = workspace
    values = (
        x,
        xs.reshape(-1),
        down,
        ds.reshape(-1),
        up,
        us.reshape(-1),
        workspace.partials,
        workspace.activation,
        workspace.alpha,
        workspace.epochs,
        out,
        out if inject is None else inject,
    )
    operands = tuple(_view(v) for v in values)
    plan.compiled(
        *operands, cutlass.Int32(plan.grid), *tma_l2_ids, *weight_priority_ids, stream
    )
    return out, inject
