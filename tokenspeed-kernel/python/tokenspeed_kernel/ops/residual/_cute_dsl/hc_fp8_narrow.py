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
    _compiled_function,
    _mixer_operands,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.narrow import (
    Fp8FusedGatedResidualKernel,
)

_PLANS = {}
_WORKSPACES = {}
_LOCK = threading.RLock()
# Phase ordinals must never reuse scratch initialized with round generations.
_EPOCH_PROTOCOL_VERSION = 8

# Token bucket to (projection tile, split count, stages).
_DOWN_TACTICS = {
    16: (64, 16, 2),
    32: (64, 16, 2),
    64: (64, 8, 2),
    128: (64, 8, 2),
    256: (64, 4, 3),
}


def _allocate_partials(kernel, rows, projection_rows, device):
    if kernel.aligned_partial_rows:
        # The 16-token aligned planes use this buffer as raw storage.
        shape = (rows, kernel.splits, projection_rows)
        return torch.empty(shape, device=device, dtype=torch.float32).transpose(0, 1)
    shape = (kernel.splits, rows, projection_rows)
    return torch.empty(shape, device=device, dtype=torch.float32)


def _activation_workspace(rows, width, device, bundle_alpha):
    if bundle_alpha:
        codes_bytes = rows * width
        buffer = torch.empty(codes_bytes + rows * 4, device=device, dtype=torch.uint8)
        activation = buffer[:codes_bytes].view(torch.float8_e4m3fn).view(rows, width)
        alpha = buffer[codes_bytes:].view(torch.float32)
        return activation, alpha
    return (
        torch.empty((rows, width), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((rows,), device=device, dtype=torch.float32),
    )


def _allocate_activation(kernel, device):
    if kernel.stamped_activation:
        records = torch.zeros(
            (kernel.n, (kernel.activation_payload_records // kernel.n + 1) * 16),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        alpha = torch.empty((kernel.n,), dtype=torch.float32, device=device)
        return records, alpha
    return _activation_workspace(
        kernel.n,
        kernel.r,
        device,
        kernel.copied_alpha,
    )


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
):
    rows = x.shape[0]
    out = torch.empty((rows, h), device=x.device, dtype=output_dtype)
    inject = (
        torch.empty((rows, g), device=x.device, dtype=output_dtype) if p > r else None
    )
    if rows == 0:
        return out, inject
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    has_injection = p == r + g
    first_bucket = token_bucket == 16
    second_bucket = token_bucket == 32
    third_bucket = token_bucket == 64
    long_tile = token_bucket == 128
    two_token_parts = token_bucket == 256
    stamped_activation = has_injection and first_bucket
    warp_owned_gates = has_injection and second_bucket
    split_activation_counter = has_injection and third_bucket

    down_m, splits, stages = _DOWN_TACTICS[token_bucket]
    down_k = 512 if two_token_parts else g * h // (splits * stages)
    down_tactic = (down_m, splits, down_k, stages)
    if stamped_activation:
        # Shorter split-K tasks make room for dedicated quantization CTAs.
        down_tactic = 64, 20, 256, 2
    down_token_tile = {64: 32, 128: 64, 256: 64}.get(token_bucket, token_bucket)
    partial_geometry = (
        second_bucket or split_activation_counter,
        first_bucket or second_bucket or split_activation_counter,
    )
    down_release = "early_arrive" if first_bucket else "native"
    down_writer_release = (1, stamped_activation, False)
    omit_down_empty = token_bucket <= 64
    entry_down_input_ring_first = token_bucket <= 64
    entry_down_prefill = True
    evict_down_weights_last = True
    input_refill_hint = (3, 4) if has_injection and two_token_parts else ()

    async_phase_arrivals = (
        "both" if first_bucket else "down" if second_bucket else "native"
    )
    quant_max_join = (
        "double"
        if two_token_parts
        else "single_active" if third_bucket else "single_full"
    )
    quant_warp_handoff = third_bucket
    quant_copy_warp = 5 if warp_owned_gates else -1
    quant_down_overlap = "early_scalar" if stamped_activation else "native"
    quant_down_poll = "acquire_fence" if stamped_activation else "native"
    quant_activation_store = (
        "pack_max" if has_injection and token_bucket <= 64 else "native"
    )
    activation_handoff = (
        "row_async_bulk"
        if stamped_activation
        else (
            "row_tags_quant" if warp_owned_gates else "warp" if third_bucket else "grid"
        )
    )
    dedicated_quant_ctas = first_bucket
    activation_retry_ns = 64 if first_bucket else 0

    up_k = 64 if long_tile else r
    up_tactic = (up_k, r // up_k)
    use_cta_reader_handoff = token_bucket <= 128
    up_register_rows = two_token_parts
    residual_issue = (
        "register_before_mma"
        if stamped_activation
        else (
            "down_entry"
            if long_tile
            else "handoff" if second_bucket or third_bucket else "up"
        )
    )
    residual_transfer = "tma" if long_tile else "lane"
    if residual_transfer == "tma" and rows >= 1 << 31:
        raise ValueError("Residual TMA coordinates must fit signed int32")
    up_hidden_columns = (
        2
        if stamped_activation
        else 4 if warp_owned_gates or split_activation_counter else 1
    )
    up_scale_issue = "up_async" if long_tile else "native"
    up_gate_exchange = "row_pairs_scalar" if split_activation_counter else "native"
    up_retirement = (
        "skip_final_and_tail_join"
        if warp_owned_gates or split_activation_counter
        else "skip_final"
    )
    up_activation_prefix = (3, True) if split_activation_counter else (0, False)
    aligned_partial_rows = stamped_activation
    pad_residual_rows = split_activation_counter
    paired_up_sigmoid = warp_owned_gates
    prefetch_up_weights_on_down_retire = split_activation_counter
    up_activation_transport = (
        "bulk"
        if stamped_activation or second_bucket or third_bucket
        else "packed_tma" if first_bucket else "native"
    )
    up_activation_delivery = (
        "bundle_deferred"
        if warp_owned_gates
        else "private_deferred" if second_bucket else "native"
    )
    omit_up_empty = True
    private_epoch_order = "coherent"
    entry_transport = "deferred"
    minimum_blocks_per_sm = 1
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
        down_token_tile=down_token_tile,
        up_tactic=up_tactic,
        use_cta_reader_handoff=use_cta_reader_handoff,
        static_partial_store=partial_geometry[0],
        static_partial_stride=partial_geometry[1],
        quant_max_join=quant_max_join,
        quant_warp_handoff=quant_warp_handoff,
        down_release=down_release,
        down_writer_release=down_writer_release,
        async_phase_arrivals=async_phase_arrivals,
        omit_down_empty=omit_down_empty,
        up_register_rows=up_register_rows,
        omit_up_empty=omit_up_empty,
        private_epoch_order=private_epoch_order,
        entry_transport=entry_transport,
        minimum_blocks_per_sm=minimum_blocks_per_sm,
        entry_down_prefill=entry_down_prefill,
        entry_down_input_ring_first=entry_down_input_ring_first,
        evict_down_weights_last=evict_down_weights_last,
        residual_issue=residual_issue,
        residual_transfer=residual_transfer,
        pad_residual_rows=pad_residual_rows,
        up_hidden_columns=up_hidden_columns,
        up_gate_exchange=up_gate_exchange,
        warp_owned_gates=warp_owned_gates,
        quant_copy_warp=quant_copy_warp,
        paired_up_sigmoid=paired_up_sigmoid,
        up_scale_issue=up_scale_issue,
        prefetch_up_weights_on_down_retire=prefetch_up_weights_on_down_retire,
        up_retirement=up_retirement,
        activation_handoff=activation_handoff,
        split_activation_counter=split_activation_counter,
        dedicated_quant_ctas=dedicated_quant_ctas,
        quant_down_overlap=quant_down_overlap,
        quant_down_poll=quant_down_poll,
        quant_activation_store=quant_activation_store,
        up_activation_prefix=up_activation_prefix,
        stamped_activation=stamped_activation,
        aligned_partial_rows=aligned_partial_rows,
        up_activation_transport=up_activation_transport,
        activation_retry_ns=activation_retry_ns,
        input_refill_hint=input_refill_hint,
        up_activation_delivery=up_activation_delivery,
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
                scratch = Workspace(
                    _allocate_partials(kernel, token_bucket, p, x.device),
                    *_allocate_activation(kernel, x.device),
                    torch.zeros(
                        (2 + kernel.extra_epoch_slots,),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                )
                operands = _mixer_operands(
                    x, xs, down, ds, up, us, scratch, out, inject
                )
                compiled = cute_ext.compile(kernel, *operands, cutlass.Int32(1), stream)
                function = _compiled_function(compiled, x.device)
                blocks = _checked(
                    driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        function,
                        kernel.threads,
                        kernel.shared_bytes,
                    )
                )
                properties = torch.cuda.get_device_properties(x.device)
                capacity = blocks * properties.multi_processor_count
                if capacity <= 0:
                    raise RuntimeError(
                        "prequantized FP8 cooperative kernel has no resident capacity"
                    )
                desired_grid = max(kernel.down_tasks, kernel.up_tasks)
                if dedicated_quant_ctas:
                    desired_grid += kernel.quant_groups
                grid = min(capacity, desired_grid)
                resources = {
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
                    "query_shared_bytes": kernel.shared_bytes,
                    "resident_capacity": capacity,
                    "blocks_per_sm": blocks,
                    "sm_count": properties.multi_processor_count,
                    "grid": grid,
                    "threads": kernel.threads,
                }
                plan = Plan(compiled, grid, kernel, resources)
                _PLANS[key] = plan
            with torch.inference_mode(False):
                workspace = Workspace(
                    _allocate_partials(plan.kernel, token_bucket, p, x.device),
                    *_allocate_activation(plan.kernel, x.device),
                    torch.zeros(
                        (plan.grid + 1 + plan.kernel.extra_epoch_slots,),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                )
            _WORKSPACES[stream_key] = workspace
    operands = _mixer_operands(x, xs, down, ds, up, us, workspace, out, inject)
    plan.compiled(*operands, cutlass.Int32(plan.grid), stream)
    return out, inject
