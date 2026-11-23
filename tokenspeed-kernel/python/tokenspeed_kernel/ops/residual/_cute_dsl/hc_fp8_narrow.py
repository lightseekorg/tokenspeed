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

# Token bucket to (projection tile, split count, K tile, stages).
_DOWN_TACTICS = {
    16: (64, 16, 320, 2),
    32: (64, 16, 320, 2),
    64: (64, 8, 640, 2),
    128: (64, 8, 640, 2),
    256: (64, 4, 512, 3),
}


def _select_residual_issue(g, h, r, p, token_bucket):
    if (g, h, r, p, token_bucket) == (4, 2560, 320, 324, 16):
        return "register_before_mma"
    if (g, h, r) == (4, 2560, 320):
        if token_bucket == 128:
            return "down_entry"
        if token_bucket in (32, 64):
            return "handoff"
    return "up"


def _select_residual_transfer(g, h, r, token_bucket):
    return "tma" if (g, h, r, token_bucket) == (4, 2560, 320, 128) else "lane"


def _select_up_retirement(g, h, r, token_bucket, has_injection):
    if has_injection and token_bucket in (32, 64):
        return "skip_final_and_tail_join"
    if (g, h, r) == (4, 2560, 320):
        return "skip_final"
    return "native"


def _select_up_hidden_columns(g, h, r, p, token_bucket):
    if (g, h, r, p) == (4, 2560, 320, 324):
        if token_bucket == 16:
            return 2
        if token_bucket in (32, 64):
            return 4
    return 1


def _select_up_gate_exchange(has_injection, token_bucket):
    return "row_pairs_scalar" if has_injection and token_bucket == 64 else "native"


def _select_quant_copy_warp(has_injection, token_bucket):
    return 5 if has_injection and token_bucket == 32 else -1


def _select_warp_owned_gates(has_injection, token_bucket):
    return has_injection and token_bucket == 32


def _select_up_scale_issue(g, h, r, token_bucket):
    return "up_async" if (g, h, r, token_bucket) == (4, 2560, 320, 128) else "native"


def _select_quant_down_overlap(has_injection, token_bucket):
    return "early_scalar" if has_injection and token_bucket == 16 else "native"


def _select_quant_down_poll(has_injection, token_bucket):
    return "acquire_fence" if has_injection and token_bucket == 16 else "native"


def _select_aligned_partial_rows(has_injection, token_bucket):
    return has_injection and token_bucket == 16


def _allocate_partials(kernel, rows, projection_rows, device):
    if kernel.aligned_partial_rows:
        # The 16-token aligned planes use this buffer as raw storage.
        shape = (rows, kernel.splits, projection_rows)
        return torch.empty(shape, device=device, dtype=torch.float32).transpose(0, 1)
    shape = (kernel.splits, rows, projection_rows)
    return torch.empty(shape, device=device, dtype=torch.float32)


def _select_quant_activation_store(has_injection, token_bucket):
    return "pack_max" if has_injection and token_bucket in (16, 32, 64) else "native"


def _select_up_activation_prefix(has_injection, token_bucket):
    return (3, True) if has_injection and token_bucket == 64 else (0, False)


def _select_activation_retry_ns(g, h, r, token_bucket):
    return 64 if (g, h, r, token_bucket) == (4, 2560, 320, 16) else 0


def _select_stage_handoff(g, h, r, p, token_bucket, async_phase_arrivals):
    if (g, h, r) == (4, 2560, 320):
        if token_bucket == 16:
            if p == r + g and async_phase_arrivals == "both":
                return "row_async_bulk", True
            return "grid", True
        if token_bucket == 32 and p == r + g and async_phase_arrivals == "down":
            return "row_tags_quant", False
        if token_bucket == 64:
            return "warp", False
    return "grid", False


def _select_down_input_refill_hint(has_injection, token_bucket):
    return (3, 4) if has_injection and token_bucket == 256 else ()


def _select_up_activation_transport(g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320):
        if token_bucket == 16:
            return "packed_tma"
        if token_bucket == 64:
            return "bulk"
    return "native"


def _select_up_resident_tactic(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 2560, 320, 128):
        return 64, 5
    k = 128 if r == 1024 else 64 if token_bucket == 128 else 320
    return k, min(3, r // k)


def _select_down_token_tile(g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320):
        return {64: 32, 128: 64, 256: 64}.get(token_bucket, token_bucket)
    return token_bucket


def _select_cta_reader_handoff(r, token_bucket):
    return r == 320 and token_bucket <= 128


def _select_partial_geometry(r, token_bucket):
    """Select fixed workspace geometry for (Down stores, ordered merge loads)."""
    if r == 320 and token_bucket == 16:
        return False, True
    if r == 320 and token_bucket == 32:
        return True, True
    return False, False


def _select_quant_max_join(g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320):
        if token_bucket in (16, 32, 128):
            return "single_full"
        if token_bucket == 64:
            return "single_active"
    return "double"


def _select_quant_warp_handoff(g, h, r, token_bucket):
    return (g, h, r, token_bucket) == (4, 2560, 320, 64)


def _select_down_release(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 2560, 320, 16):
        return "early_arrive"
    return "native"


def _select_down_writer_release(has_injection, token_bucket):
    return 1, has_injection and token_bucket == 16, False


def _select_async_phase_arrivals(device, g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320) and torch.cuda.get_device_capability(device)[
        0
    ] == 10:
        if token_bucket == 16:
            return "both"
        if token_bucket == 32:
            return "down"
    return "native"


def _select_omit_down_empty(g, h, r, token_bucket):
    return (g, h, r) == (4, 2560, 320) and token_bucket in (16, 32, 64)


def _select_up_register_rows(g, h, r, token_bucket):
    return (g, h, r, token_bucket) == (4, 2560, 320, 256)


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
    down_tactic = _DOWN_TACTICS[token_bucket]
    if has_injection and token_bucket == 16:
        # Fit 120 shorter Down tasks and 16 dedicated quantization CTAs in one resident grid.
        down_tactic = 64, 20, 256, 2
    down_token_tile = _select_down_token_tile(g, h, r, token_bucket)
    up_tactic = _select_up_resident_tactic(g, h, r, token_bucket)
    use_cta_reader_handoff = _select_cta_reader_handoff(r, token_bucket)
    partial_geometry = _select_partial_geometry(r, token_bucket)
    if (g, h, r, p, token_bucket) == (4, 2560, 320, 324, 64):
        partial_geometry = True, True
    quant_max_join = _select_quant_max_join(g, h, r, token_bucket)
    quant_warp_handoff = _select_quant_warp_handoff(g, h, r, token_bucket)
    down_release = _select_down_release(g, h, r, token_bucket)
    async_phase_arrivals = _select_async_phase_arrivals(x.device, g, h, r, token_bucket)
    omit_down_empty = _select_omit_down_empty(g, h, r, token_bucket)
    up_register_rows = _select_up_register_rows(g, h, r, token_bucket)
    # These launch settings apply to every SM103 narrow bucket.
    omit_up_empty = True
    use_native_warp_max = True
    private_epoch_order = "coherent"
    entry_transport = "deferred"
    minimum_blocks_per_sm = 1
    entry_down_prefill = True
    entry_down_input_ring_first = token_bucket <= 64
    evict_down_weights_last = True
    residual_issue = _select_residual_issue(g, h, r, p, token_bucket)
    residual_transfer = _select_residual_transfer(g, h, r, token_bucket)
    if residual_transfer == "tma" and rows >= 1 << 31:
        raise ValueError("Residual TMA coordinates must fit signed int32")
    up_hidden_columns = _select_up_hidden_columns(g, h, r, p, token_bucket)
    up_scale_issue = _select_up_scale_issue(g, h, r, token_bucket)
    activation_handoff, dedicated_quant_ctas = _select_stage_handoff(
        g, h, r, p, token_bucket, async_phase_arrivals
    )
    down_writer_release = _select_down_writer_release(has_injection, token_bucket)
    up_gate_exchange = _select_up_gate_exchange(has_injection, token_bucket)
    warp_owned_gates = _select_warp_owned_gates(has_injection, token_bucket)
    quant_copy_warp = _select_quant_copy_warp(has_injection, token_bucket)
    up_retirement = _select_up_retirement(g, h, r, token_bucket, has_injection)
    quant_down_overlap = _select_quant_down_overlap(has_injection, token_bucket)
    quant_down_poll = _select_quant_down_poll(has_injection, token_bucket)
    quant_activation_store = _select_quant_activation_store(has_injection, token_bucket)
    up_activation_prefix = _select_up_activation_prefix(has_injection, token_bucket)
    stamped_activation = has_injection and token_bucket == 16
    aligned_partial_rows = _select_aligned_partial_rows(has_injection, token_bucket)
    pad_residual_rows = has_injection and token_bucket == 64
    paired_up_sigmoid = has_injection and token_bucket == 32
    input_refill_hint = _select_down_input_refill_hint(has_injection, token_bucket)
    split_activation_counter = has_injection and token_bucket == 64
    prefetch_up_weights_on_down_retire = has_injection and token_bucket == 64
    up_activation_transport = _select_up_activation_transport(g, h, r, token_bucket)
    activation_retry_ns = _select_activation_retry_ns(g, h, r, token_bucket)
    if activation_handoff == "row_tags_quant":
        up_activation_delivery = "bundle_deferred"
    elif token_bucket == 32:
        up_activation_delivery = "private_deferred"
    else:
        up_activation_delivery = "native"
    if up_activation_delivery != "native" or activation_handoff == "row_async_bulk":
        up_activation_transport = "bulk"
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
        down_token_tile,
        up_tactic,
        use_cta_reader_handoff,
        partial_geometry,
        quant_max_join,
        quant_warp_handoff,
        down_release,
        down_writer_release,
        async_phase_arrivals,
        omit_down_empty,
        up_register_rows,
        omit_up_empty,
        use_native_warp_max,
        private_epoch_order,
        entry_transport,
        minimum_blocks_per_sm,
        entry_down_prefill,
        entry_down_input_ring_first,
        evict_down_weights_last,
        residual_issue,
        residual_transfer,
        pad_residual_rows,
        up_hidden_columns,
        up_gate_exchange,
        warp_owned_gates,
        quant_copy_warp,
        paired_up_sigmoid,
        up_scale_issue,
        prefetch_up_weights_on_down_retire,
        up_retirement,
        activation_handoff,
        split_activation_counter,
        dedicated_quant_ctas,
        quant_down_overlap,
        quant_down_poll,
        quant_activation_store,
        up_activation_prefix,
        stamped_activation,
        aligned_partial_rows,
        up_activation_transport,
        activation_retry_ns,
        input_refill_hint,
        up_activation_delivery,
        _EPOCH_PROTOCOL_VERSION,
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
                use_native_warp_max=use_native_warp_max,
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
